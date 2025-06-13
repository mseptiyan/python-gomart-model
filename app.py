from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model Apriori dari file pickle
with open('apriori_market_basket_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

association_rules = model_data['association_rules']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_items = data.get('keranjang', [])

    rekomendasi = []

    for _, rule in association_rules.iterrows():
        antecedents = set(rule['antecedents'])  # barang yang harus ada di keranjang
        consequents = set(rule['consequents'])  # barang yang direkomendasikan

        # Jika semua antecedents ada di input user
        if antecedents.issubset(set(input_items)):
            for item in consequents:
                # Hindari merekomendasikan barang yang sudah ada di keranjang
                if item not in input_items:
                    rekomendasi.append({
                        'Nama Barang': item,
                        'Kuantitas': 1,
                        'Harga': 0  # Ganti kalau kamu punya info harga
                    })

    return jsonify({'rekomendasi': rekomendasi})

if __name__ == '__main__':
    app.run(debug=True)
