import streamlit as st
import h5py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Keranjang Belanja GoMart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

class H5MarketBasketRecommender:
    """
    Rekomendasi Market Basket dari file H5 (Apriori/FP-Growth)
    """

    def __init__(self, h5_model_path):
        self.model_path = h5_model_path
        self.load_model()

    def load_model(self):
        with h5py.File(self.model_path, 'r') as f:
            self.metadata = dict(f['model_metadata'].attrs)
            # Frequent itemsets
            if 'frequent_itemsets' in f:
                freq_group = f['frequent_itemsets']
                self.frequent_itemsets = {
                    'itemsets': [eval(s.decode('utf-8')) for s in freq_group['itemsets'][:]],
                    'support': freq_group['support'][:],
                    'length': freq_group['length'][:]
                }
            # Association rules
            if 'association_rules' in f:
                rules_group = f['association_rules']
                self.association_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in rules_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in rules_group['consequents'][:]],
                    'support': rules_group['support'][:],
                    'confidence': rules_group['confidence'][:],
                    'lift': rules_group['lift'][:]
                }
            # Cross-selling rules
            if 'cross_selling_rules' in f and len(f['cross_selling_rules'].keys()) > 0:
                cross_group = f['cross_selling_rules']
                self.cross_selling_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in cross_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in cross_group['consequents'][:]],
                    'support': cross_group['support'][:],
                    'confidence': cross_group['confidence'][:],
                    'lift': cross_group['lift'][:]
                }
            else:
                self.cross_selling_rules = None
            # Upselling rules
            if 'upselling_rules' in f and len(f['upselling_rules'].keys()) > 0:
                up_group = f['upselling_rules']
                self.upselling_rules = {
                    'antecedents': [eval(s.decode('utf-8')) for s in up_group['antecedents'][:]],
                    'consequents': [eval(s.decode('utf-8')) for s in up_group['consequents'][:]],
                    'support': up_group['support'][:],
                    'confidence': up_group['confidence'][:],
                    'lift': up_group['lift'][:]
                }
            else:
                self.upselling_rules = None
            # Item statistics
            if 'item_statistics' in f:
                items_group = f['item_statistics']
                self.item_names = [s.decode('utf-8') for s in items_group['item_names'][:]]
                self.item_frequencies = items_group['item_frequencies'][:]
                self.all_items = [s.decode('utf-8') for s in items_group['all_items'][:]]
            # Performance metrics
            if 'performance_metrics' in f:
                self.performance_metrics = dict(f['performance_metrics'].attrs)

    def get_cross_selling_recommendations(self, item, top_n=5, min_confidence=0.3):
        if not self.cross_selling_rules:
            return []
        recommendations = []
        for i, antecedent in enumerate(self.cross_selling_rules['antecedents']):
            if item in antecedent and self.cross_selling_rules['confidence'][i] >= min_confidence:
                consequent = self.cross_selling_rules['consequents'][i]
                if len(consequent) == 1:
                    recommendations.append({
                        'recommended_item': consequent[0],
                        'confidence': float(self.cross_selling_rules['confidence'][i]),
                        'support': float(self.cross_selling_rules['support'][i]),
                        'lift': float(self.cross_selling_rules['lift'][i])
                    })
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_n]

    def get_upselling_recommendations(self, item, top_n=5, min_confidence=0.25):
        if not self.upselling_rules:
            return []
        recommendations = []
        for i, antecedent in enumerate(self.upselling_rules['antecedents']):
            if item in antecedent and self.upselling_rules['confidence'][i] >= min_confidence:
                consequents = self.upselling_rules['consequents'][i]
                recommendations.append({
                    'recommended_items': consequents,
                    'confidence': float(self.upselling_rules['confidence'][i]),
                    'support': float(self.upselling_rules['support'][i]),
                    'lift': float(self.upselling_rules['lift'][i]),
                    'bundle_size': len(consequents)
                })
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_n]

    def get_basket_recommendations(self, basket_items, top_n=10, min_confidence=0.2):
        all_recommendations = {}
        for item in basket_items:
            if item in self.all_items:
                cross_recs = self.get_cross_selling_recommendations(item, top_n=20, min_confidence=min_confidence)
                for rec in cross_recs:
                    rec_item = rec['recommended_item']
                    if rec_item not in basket_items:
                        if rec_item not in all_recommendations:
                            all_recommendations[rec_item] = {
                                'total_confidence': 0,
                                'total_support': 0,
                                'total_lift': 0,
                                'rule_count': 0,
                                'supporting_items': []
                            }
                        all_recommendations[rec_item]['total_confidence'] += rec['confidence']
                        all_recommendations[rec_item]['total_support'] += rec['support']
                        all_recommendations[rec_item]['total_lift'] += rec['lift']
                        all_recommendations[rec_item]['rule_count'] += 1
                        all_recommendations[rec_item]['supporting_items'].append(item)
        final_recommendations = []
        for item, stats in all_recommendations.items():
            if stats['rule_count'] > 0:
                final_recommendations.append({
                    'recommended_item': item,
                    'avg_confidence': stats['total_confidence'] / stats['rule_count'],
                    'avg_support': stats['total_support'] / stats['rule_count'],
                    'avg_lift': stats['total_lift'] / stats['rule_count'],
                    'supporting_rules': stats['rule_count'],
                    'supporting_items': stats['supporting_items']
                })
        final_recommendations.sort(key=lambda x: x['avg_confidence'], reverse=True)
        return final_recommendations[:top_n]

    def get_model_info(self):
        return {
            'model_metadata': self.metadata,
            'performance_metrics': self.performance_metrics if hasattr(self, 'performance_metrics') else {},
            'total_items': len(self.all_items) if hasattr(self, 'all_items') else 0,
            'top_items': dict(zip(self.item_names[:10], self.item_frequencies[:10])) if hasattr(self, 'item_names') else {}
        }

@st.cache_resource
def load_model(model_path):
    try:
        return H5MarketBasketRecommender(model_path)
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None

def main():
    st.title("🛒 Dashboard Analisis Keranjang Belanja GoMart")
    st.markdown("---")
    st.sidebar.title("📊 Navigasi")

    # Input path model
    model_path = st.sidebar.text_input(
        "Lokasi File Model H5",
        value="market_basket_model.h5",
        help="Masukkan path ke file model H5 Anda"
    )

    if st.sidebar.button("🔄 Muat Model"):
        st.session_state.model = load_model(model_path)
        if st.session_state.model:
            st.sidebar.success("✅ Model berhasil dimuat!")
        else:
            st.sidebar.error("❌ Gagal memuat model")

    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)

    if st.session_state.model is None:
        st.error("⚠️ Pastikan file model tersedia lalu klik 'Muat Model'.")
        return

    model = st.session_state.model

    page = st.sidebar.selectbox(
        "Pilih Analisis",
        ["📈 Ringkasan Model", "🎯 Rekomendasi Satu Barang", "🛍️ Rekomendasi Keranjang", "📊 Analisis Lanjutan"]
    )

    if page == "📈 Ringkasan Model":
        show_model_overview(model)
    elif page == "🎯 Rekomendasi Satu Barang":
        show_single_item_recommendations(model)
    elif page == "🛍️ Rekomendasi Keranjang":
        show_basket_recommendations(model)
    elif page == "📊 Analisis Lanjutan":
        show_advanced_analytics(model)

def show_model_overview(model):
    st.header("📈 Ringkasan Model")
    model_info = model.get_model_info()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Barang", model_info['total_items'])
    with col2:
        st.metric("Total Aturan", model_info['model_metadata'].get('total_association_rules', 0))
    with col3:
        st.metric("Total Transaksi", model_info['model_metadata'].get('total_transactions', 0))
    with col4:
        avg_conf = model_info['performance_metrics'].get('avg_confidence', 0)
        st.metric("Rata-rata Confidence", f"{avg_conf:.3f}")

    st.subheader("🔧 Konfigurasi Model")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.info(f"**Algoritma:** {model_info['model_metadata'].get('algorithm', '-')}")
        st.info(f"**Min Support:** {model_info['model_metadata'].get('min_support_threshold', '-')}")
        st.info(f"**Min Confidence:** {model_info['model_metadata'].get('min_confidence_threshold', '-')}")
    with config_col2:
        st.info(f"**Dibuat:** {model_info['model_metadata'].get('created_at', '-')[:19]}")
        st.info(f"**Framework:** {model_info['model_metadata'].get('framework_version', '-')}")
        st.info(f"**Frequent Itemsets:** {model_info['model_metadata'].get('total_frequent_itemsets', 0)}")

    st.subheader("🏆 10 Barang Terpopuler")
    if model_info['top_items']:
        items_df = pd.DataFrame(list(model_info['top_items'].items()), columns=['Barang', 'Frekuensi'])
        fig = px.bar(
            items_df,
            x='Frekuensi',
            y='Barang',
            orientation='h',
            color='Frekuensi',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    if 'performance_metrics' in model_info and model_info['performance_metrics']:
        st.subheader("📊 Metrik Performa")
        perf_metrics = model_info['performance_metrics']
        metrics_df = pd.DataFrame({
            'Metrik': ['Support', 'Confidence', 'Lift'],
            'Rata-rata': [
                perf_metrics.get('avg_support', 0),
                perf_metrics.get('avg_confidence', 0),
                perf_metrics.get('avg_lift', 0)
            ],
            'Maksimum': [
                perf_metrics.get('max_support', 0),
                perf_metrics.get('max_confidence', 0),
                perf_metrics.get('max_lift', 0)
            ],
            'Minimum': [
                perf_metrics.get('min_support', 0),
                perf_metrics.get('min_confidence', 0),
                perf_metrics.get('min_lift', 0)
            ]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Rata-rata', x=metrics_df['Metrik'], y=metrics_df['Rata-rata']))
        fig.add_trace(go.Bar(name='Maksimum', x=metrics_df['Metrik'], y=metrics_df['Maksimum']))
        fig.add_trace(go.Bar(name='Minimum', x=metrics_df['Metrik'], y=metrics_df['Minimum']))
        fig.update_layout(title="Metrik Performa Aturan Asosiasi", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_single_item_recommendations(model):
    st.header("🎯 Rekomendasi Pembelian Bersamaan (Satu Barang)")
    available_items = model.all_items if hasattr(model, 'all_items') else []
    if not available_items:
        st.error("Tidak ada data barang pada model.")
        return

    selected_item = st.selectbox(
        "Pilih barang untuk melihat rekomendasi pembelian bersamaan:",
        available_items,
        help="Pilih satu barang, lalu lihat apa saja barang lain yang sering dibeli bersamaan."
    )

    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Batas Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Semakin tinggi nilai confidence, semakin kuat hubungan antar barang."
        )
    with col2:
        top_n = st.slider(
            "Jumlah Rekomendasi Ditampilkan",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )

    if st.button("🔍 Tampilkan Rekomendasi"):
        cross_recs = model.get_cross_selling_recommendations(
            selected_item,
            top_n=top_n,
            min_confidence=confidence_threshold
        )
        up_recs = model.get_upselling_recommendations(
            selected_item,
            top_n=top_n,
            min_confidence=confidence_threshold-0.05
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔄 Rekomendasi Barang untuk Dibeli Bersamaan")
            if cross_recs:
                for rec in cross_recs:
                    st.success(
                        f"Jika Anda membeli **{selected_item}**, sistem merekomendasikan juga membeli **{rec['recommended_item']}** "
                        f"(Confidence: {rec['confidence']:.2f}, Support: {rec['support']:.2f}, Lift: {rec['lift']:.2f})"
                    )
            else:
                st.info("Belum ada rekomendasi cross-selling untuk barang ini pada threshold saat ini.")

        with col2:
            st.subheader("📈 Rekomendasi Paket (Upselling)")
            if up_recs:
                for i, rec in enumerate(up_recs[:5]):
                    st.info(
                        f"Jika Anda membeli **{selected_item}**, pertimbangkan juga paket: **{', '.join(rec['recommended_items'])}** "
                        f"(Confidence: {rec['confidence']:.2f}, Support: {rec['support']:.2f}, Lift: {rec['lift']:.2f})"
                    )
            else:
                st.info("Belum ada rekomendasi upselling untuk barang ini pada threshold saat ini.")

def show_basket_recommendations(model):
    st.header("🛍️ Rekomendasi Berdasarkan Keranjang Belanja Anda")
    available_items = model.all_items if hasattr(model, 'all_items') else []
    if not available_items:
        st.error("Tidak ada data barang pada model.")
        return

    st.subheader("🛒 Tambahkan Barang ke Keranjang Anda")
    selected_basket = st.multiselect(
        "Pilih barang yang sudah ada di keranjang:",
        available_items,
        help="Pilih beberapa barang yang sudah Anda masukkan ke keranjang."
    )

    if not selected_basket:
        st.info("Silakan tambahkan barang ke keranjang untuk mendapatkan rekomendasi.")
        return

    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Batas Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="basket_confidence"
        )
    with col2:
        top_n = st.slider(
            "Jumlah Rekomendasi Ditampilkan",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="basket_top_n"
        )

    st.subheader("📋 Barang di Keranjang Anda")
    basket_df = pd.DataFrame({'Barang di Keranjang': selected_basket})
    st.dataframe(basket_df, use_container_width=True)

    if st.button("🎯 Tampilkan Rekomendasi untuk Keranjang"):
        recommendations = model.get_basket_recommendations(
            selected_basket,
            top_n=top_n,
            min_confidence=confidence_threshold
        )

        if recommendations:
            st.subheader("✨ Barang yang Direkomendasikan untuk Dibeli Bersamaan")
            for rec in recommendations:
                st.success(
                    f"Barang **{rec['recommended_item']}** direkomendasikan untuk dibeli bersama "
                    f"{', '.join(rec['supporting_items'])} "
                    f"(Confidence rata-rata: {rec['avg_confidence']:.2f})"
                )
        else:
            st.info("Belum ada rekomendasi untuk kombinasi keranjang ini pada threshold saat ini.")

def show_advanced_analytics(model):
    st.header("📊 Analisis Lanjutan Sistem Rekomendasi Penjualan GoMart")

    # Analisis aturan asosiasi
    if hasattr(model, 'association_rules'):
        st.subheader("📈 Analisis Aturan Rekomendasi (Market Basket Analysis)")
        rules_data = []
        for i in range(len(model.association_rules['antecedents'])):
            rules_data.append({
                'Barang Awal (Antecedents)': ', '.join(model.association_rules['antecedents'][i]),
                'Barang Direkomendasikan (Consequents)': ', '.join(model.association_rules['consequents'][i]),
                'Support': round(model.association_rules['support'][i], 3),
                'Confidence': round(model.association_rules['confidence'][i], 3),
                'Lift': round(model.association_rules['lift'][i], 3)
            })
        rules_df = pd.DataFrame(rules_data)
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Min Support", 0.0, 1.0, 0.01, 0.01)
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.2, 0.01)
        with col3:
            min_lift = st.slider("Min Lift", 0.0, 10.0, 1.0, 0.1)
        filtered_rules = rules_df[
            (rules_df['Support'] >= min_support) &
            (rules_df['Confidence'] >= min_confidence) &
            (rules_df['Lift'] >= min_lift)
        ]
        st.write(f"Menampilkan {len(filtered_rules)} aturan rekomendasi dari total {len(rules_df)} aturan.")
        if len(filtered_rules) > 0:
            st.dataframe(filtered_rules.head(20), use_container_width=True)
            fig = px.scatter(
                filtered_rules,
                x='Support',
                y='Confidence',
                color='Lift',
                size='Lift',
                hover_data=['Barang Awal (Antecedents)', 'Barang Direkomendasikan (Consequents)'],
                title="Aturan Rekomendasi: Support vs Confidence (warna: Lift)"
            )
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(
                    filtered_rules,
                    x='Confidence',
                    nbins=20,
                    title="Distribusi Confidence Aturan Rekomendasi"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_hist2 = px.histogram(
                    filtered_rules,
                    x='Lift',
                    nbins=20,
                    title="Distribusi Lift Aturan Rekomendasi"
                )
                st.plotly_chart(fig_hist2, use_container_width=True)
        else:
            st.info("Tidak ada aturan rekomendasi yang memenuhi filter.")

    # Analisis frekuensi item
    if hasattr(model, 'item_names') and hasattr(model, 'item_frequencies'):
        st.subheader("📊 Analisis Frekuensi Penjualan Produk")
        freq_df = pd.DataFrame({
            'Produk': model.item_names,
            'Frekuensi': model.item_frequencies
        })
        top_n_items = st.slider("Tampilkan Top N Produk Terlaris", 5, 50, 20)
        top_items_df = freq_df.head(top_n_items)
        fig = px.bar(
            top_items_df,
            x='Frekuensi',
            y='Produk',
            orientation='h',
            title=f"Top {top_n_items} Produk Terlaris di GoMart"
        )
        fig.update_layout(height=max(400, top_n_items * 20))
        st.plotly_chart(fig, use_container_width=True)
        fig_dist = px.histogram(
            freq_df,
            x='Frekuensi',
            nbins=30,
            title="Distribusi Frekuensi Penjualan Produk"
        )
        st.plotly_chart(fig_dist, use_container_width=True)


if __name__ == "__main__":
    main()
