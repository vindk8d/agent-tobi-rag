#!/usr/bin/env python3

import asyncio
import tempfile
import os

async def debug_chunking():
    # Copy the Ford Bronco content to a temp file (same as test)
    ford_bronco_content = """# Ford Bronco Outer Banks with Sasquatch™ Package

## Technical Specifications

-   **Engine**: 2.7L EcoBoost® V6
    -   Maximum Power: 335 PS @ 5,570 rpm
    -   Maximum Torque: 555 Nm @ 3,000 rpm
-   **Transmission**: 10-Speed Automatic Transmission
-   **4x4 System**: Advanced 4x4 with Automatic On-Demand Engagement

## Exterior

-   Carbonized Gray Molded-In-Color Hard Top with Sound Deadening
    Headliner
-   Removable Roof and Doors
-   LED Headlights with LED Signature Lighting

## Interior and Convenience

-   8-Inch Digital Instrument Cluster
-   Keyless Entry with Push Button Start
-   Electric Parking Brake

## Safety and Security

-   AdvanceTrac® with RSC® (Roll Stability Control™)
-   Airbags: Front (Driver & Passenger), Side (Front & Side), Safety
    Canopy®
-   Rear View Camera with Rollover Sensor
"""

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(ford_bronco_content)
        temp_file = f.name

    try:
        # Test the actual document loading and chunking pipeline
        from backend.rag.document_loader import DocumentLoader, split_documents
        
        print("🔍 Step 1: Loading document...")
        docs = DocumentLoader.load_from_file(temp_file, 'markdown')
        print(f"   📄 Loaded {len(docs)} documents")
        if docs:
            print(f"   📝 Content length: {len(docs[0].page_content)} characters")
            print(f"   📝 Content preview: {docs[0].page_content[:200]}...")
        
        print("\n🔍 Step 2: Testing section-based chunking...")
        chunks = await split_documents(docs, chunking_method="section_based")
        print(f"   📊 Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i}:")
            print(f"     Length: {len(chunk.page_content)} chars")
            print(f"     Preview: {chunk.page_content[:100]}...")
            print(f"     Metadata: {chunk.metadata}")
        
        print("\n🔍 Step 3: Testing recursive chunking for comparison...")
        recursive_chunks = await split_documents(docs, chunking_method="recursive")
        print(f"   📊 Created {len(recursive_chunks)} recursive chunks")
        
    finally:
        # Clean up temp file
        os.unlink(temp_file)

if __name__ == "__main__":
    asyncio.run(debug_chunking())
