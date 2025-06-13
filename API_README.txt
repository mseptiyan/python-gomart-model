# API Rekomendasi GoMart

Endpoint:
POST http://127.0.0.1:5000/predict

Request (JSON):
{
  "keranjang": ["Susu", "Roti"]
}

Response (JSON):
{
  "rekomendasi": [
    {"Nama Barang": "Roti", "Kuantitas": 1, "Harga": 0}
  ]
}

Catatan:
- Label output sudah sesuai kebutuhan dashboard.
- Jika butuh field tambahan, bisa request ke backend.
- Backend berjalan di lokal, jika ingin diakses online harus dihosting.
