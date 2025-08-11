import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'RAG-Tobi Frontend',
    timestamp: new Date().toISOString(),
  });
}