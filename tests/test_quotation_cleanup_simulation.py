import os
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import requests

from core.database import db_client
from core.storage import upload_quotation_pdf, create_signed_quotation_url
from backend.monitoring.quotation_cleanup import cleanup_expired_quotations


pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set to run storage integration tests",
)


async def _get_or_create_customer_id() -> str:
    client = db_client.client
    existing = client.table("customers").select("id").limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    inserted = client.table("customers").insert({
        "name": "Test Customer",
        "email": f"test.customer+{datetime.now().timestamp()}@example.com",
        "phone": "09170000000",
    }).execute()
    return inserted.data[0]["id"]


async def _get_or_create_employee_id() -> str:
    client = db_client.client
    existing = client.table("employees").select("id").limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    inserted = client.table("employees").insert({
        "name": "Test Employee",
        "email": f"employee.test+{datetime.now().timestamp()}@example.com",
        "department": "sales",
        "role_title": "sales_agent",
    }).execute()
    return inserted.data[0]["id"]


def _mk_pdf_bytes() -> bytes:
    # Use a minimal valid PDF header with a small body to keep it simple
    return (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
            b"2 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n"
            b"xref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n"
            b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n")


async def _insert_quotation(
    client,
    *,
    customer_id: str,
    employee_id: str,
    pdf_path: str,
    status: str,
    created_at: datetime,
    expires_at: datetime,
) -> str:
    # Construct a dummy URL containing '/quotations/' so cleanup can extract the path
    pdf_url = f"https://example.local/storage/v1/object/sign/quotations/{pdf_path}?token=dummy"
    data = {
        "customer_id": customer_id,
        "employee_id": employee_id,
        "vehicle_specs": {},
        "pricing_data": {"base_price": 1000, "total": 1000},
        "pdf_url": pdf_url,
        "pdf_filename": pdf_path.split("/")[-1],
        # Let trigger generate quotation_number
        "expires_at": expires_at.isoformat(),
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "status": status,
        "title": "Test Quotation",
    }
    res = client.table("quotations").insert(data).execute()
    return res.data[0]["id"]


async def _get_quotation(client, quotation_id: str) -> dict:
    res = client.table("quotations").select("id,pdf_url,status,expires_at").eq("id", quotation_id).execute()
    return res.data[0] if res.data else {}


@pytest.mark.asyncio
async def test_upload_and_cleanup_behaviour():
    client = db_client.client

    # Arrange: ensure we have related entities
    customer_id, employee_id = await asyncio.gather(
        _get_or_create_customer_id(), _get_or_create_employee_id()
    )

    # Upload target file to be deleted (expired, non-accepted)
    uploaded = await upload_quotation_pdf(
        _mk_pdf_bytes(), customer_id=customer_id, employee_id=employee_id
    )
    now = datetime.now(timezone.utc)

    delete_qid = await _insert_quotation(
        client,
        customer_id=customer_id,
        employee_id=employee_id,
        pdf_path=uploaded["path"],
        status="sent",
        created_at=now - timedelta(days=2),
        expires_at=now - timedelta(days=1),
    )

    # Control A: accepted (should NOT delete even if expired)
    uploaded_keep_accepted = await upload_quotation_pdf(
        _mk_pdf_bytes(), customer_id=customer_id, employee_id=employee_id
    )
    keep_acc_qid = await _insert_quotation(
        client,
        customer_id=customer_id,
        employee_id=employee_id,
        pdf_path=uploaded_keep_accepted["path"],
        status="accepted",
        created_at=now - timedelta(days=2),
        expires_at=now - timedelta(days=1),
    )

    # Control B: future expiry (should NOT delete)
    uploaded_keep_future = await upload_quotation_pdf(
        _mk_pdf_bytes(), customer_id=customer_id, employee_id=employee_id
    )
    keep_future_qid = await _insert_quotation(
        client,
        customer_id=customer_id,
        employee_id=employee_id,
        pdf_path=uploaded_keep_future["path"],
        status="sent",
        created_at=now,
        expires_at=now + timedelta(days=7),
    )

    # Control C: malformed pdf_url (skip deletion)
    uploaded_malformed = await upload_quotation_pdf(
        _mk_pdf_bytes(), customer_id=customer_id, employee_id=employee_id
    )
    malformed_pdf_url = "https://example.local/not-quotations/path.pdf"
    res = client.table("quotations").insert({
        "customer_id": customer_id,
        "employee_id": employee_id,
        "vehicle_specs": {},
        "pricing_data": {"base_price": 1000, "total": 1000},
        "pdf_url": malformed_pdf_url,
        "pdf_filename": "path.pdf",
        "created_at": (now - timedelta(days=2)).isoformat(),
        "updated_at": (now - timedelta(days=2)).isoformat(),
        "expires_at": (now - timedelta(days=1)).isoformat(),
        "status": "sent",
        "title": "Malformed URL Quotation",
    }).execute()
    malformed_qid = res.data[0]["id"]

    # Verify targets exist before cleanup by fetching signed URLs and doing a GET
    signed_delete = await create_signed_quotation_url(uploaded["path"], 60)
    r = requests.get(signed_delete)
    assert r.status_code == 200

    # Act: run cleanup
    summary = cleanup_expired_quotations(max_delete=200)
    assert summary["selected"] >= 1

    # Assert: target was deleted and DB updated
    post = await _get_quotation(client, delete_qid)
    assert post["status"] == "expired"
    assert post["pdf_url"] is None

    # Signed URL should now 404 or be inaccessible
    inaccessible = False
    try:
        signed_delete_after = await create_signed_quotation_url(uploaded["path"], 60)
        r2 = requests.get(signed_delete_after)
        inaccessible = r2.status_code in (403, 404)
    except Exception:
        # If the object no longer exists, signing may fail which is acceptable
        inaccessible = True
    assert inaccessible

    # Assert: accepted remains
    acc_row = await _get_quotation(client, keep_acc_qid)
    assert acc_row["status"] == "accepted"
    assert acc_row["pdf_url"] is not None
    signed_keep_acc = await create_signed_quotation_url(uploaded_keep_accepted["path"], 60)
    r3 = requests.get(signed_keep_acc)
    assert r3.status_code == 200

    # Assert: future expiry remains
    fut_row = await _get_quotation(client, keep_future_qid)
    assert fut_row["status"] == "sent"
    assert fut_row["pdf_url"] is not None
    signed_keep_fut = await create_signed_quotation_url(uploaded_keep_future["path"], 60)
    r4 = requests.get(signed_keep_fut)
    assert r4.status_code == 200

    # Assert: malformed pdf_url unchanged (not deleted)
    mal_row = await _get_quotation(client, malformed_qid)
    assert mal_row["pdf_url"] == malformed_pdf_url


