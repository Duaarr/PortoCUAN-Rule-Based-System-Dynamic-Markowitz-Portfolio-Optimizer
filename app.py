"""
app.py

Pintu masuk aplikasi Flask PortoCUAN.

Alur aplikasi:
1. /          : halaman utama
2. /input     : input dana, profil risiko, dan sektor
3. /screening : Rule-Based System untuk menentukan saham lolos/tidak lolos
4. /optimasi  : Dynamic Markowitz pada window 12 bulan terakhir
5. /backtest  : simulasi historis dengan rolling window dan rebalancing bulanan
6. /hasil     : ringkasan final rekomendasi portofolio

Versi terbaru:
- Metadata data historis dipisahkan dari tanggal rebalancing.
- Backtesting menampilkan jadwal rebalancing yang jelas.
"""

import json
import traceback

from flask import Flask, render_template, request, session, redirect, url_for

from logic.data_fetcher import (
    SEKTOR_SAHAM,
    ambil_data_saham,
    get_saham_dari_sektor,
    get_semua_saham,
    hitung_return_harian,
)

from logic.rbs import jalankan_rbs
from logic.rolling import ambil_window_terakhir

from logic.markowitz import (
    INFO_SAHAM,
    format_hasil_untuk_tampilan,
    hitung_efficient_frontier,
    optimasi_portofolio_qp,
)

from logic.backtest import (
    jalankan_backtest,
    format_backtest_untuk_chart,
)


app = Flask(__name__)
app.secret_key = 'skripsi-portofolio-2026'


# Konstanta utama penelitian.
TAHUN_DATA = 5
WINDOW_BULAN = 12
STEP_BULAN = 1
MIN_SAHAM_LOLOS = 3


_cache_data = {}


def _get_data_harga(tickers: list[str]):
    """
    Cache data supaya Yahoo Finance tidak dipanggil berulang-ulang.
    Kalau ticker sama, data lama dipakai lagi.
    """

    key = tuple(sorted(tickers))

    if key not in _cache_data:
        df = ambil_data_saham(tickers, tahun=TAHUN_DATA)

        if not df.empty:
            _cache_data[key] = df

        return df

    return _cache_data[key]


def _metadata_data(data_harga):
    """
    Membuat metadata periode data untuk UI.

    Metadata ini menjelaskan:
    - data historis mulai dari tanggal berapa,
    - data historis selesai di tanggal berapa,
    - jumlah hari perdagangan,
    - jumlah saham valid.

    Ini berbeda dari tanggal rebalancing.
    """

    if data_harga is None or data_harga.empty:
        return {
            'tahun_data': TAHUN_DATA,
            'data_mulai': '-',
            'data_selesai': '-',
            'jumlah_hari': 0,
            'jumlah_saham_valid': 0,
        }

    return {
        'tahun_data': TAHUN_DATA,
        'data_mulai': str(data_harga.index[0].date()),
        'data_selesai': str(data_harga.index[-1].date()),
        'jumlah_hari': len(data_harga),
        'jumlah_saham_valid': data_harga.shape[1],
    }


def _batas_bobot(profil_risiko: str):
    """
    Constraint bobot berdasarkan profil risiko.
    Semakin tinggi profil risiko, semakin besar bobot maksimum per saham.
    """

    batas = {
        'Rendah': (0.05, 0.35),
        'Sedang': (0.05, 0.40),
        'Tinggi': (0.05, 0.50),
    }

    return batas.get(profil_risiko, batas['Sedang'])


def _kategori_risiko(expected_risk_pct: float) -> str:
    """
    Memberi label risiko berdasarkan volatilitas portofolio.
    """

    if expected_risk_pct < 10:
        return 'Konservatif'

    if expected_risk_pct < 20:
        return 'Moderat'

    return 'Agresif'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input', methods=['GET', 'POST'])
def input_data():
    if request.method == 'GET':
        return render_template(
            'input.html',
            sektor_list=list(SEKTOR_SAHAM.keys())
        )

    session['dana_investasi'] = float(
        request.form.get('dana_investasi', 2_000_000)
    )
    session['profil_risiko'] = request.form.get('profil_risiko', 'Sedang')
    session['sektor_terpilih'] = request.form.getlist('sektor')

    # Jika user tidak memilih sektor, sistem memakai semua sektor.
    if not session['sektor_terpilih']:
        session['sektor_terpilih'] = list(SEKTOR_SAHAM.keys())

    # Bersihkan hasil lama supaya percobaan baru tidak tercampur.
    session.pop('hasil_rbs_json', None)
    session.pop('hasil_optimasi_json', None)
    session.pop('backtest_json', None)
    session.pop('saham_lolos', None)

    _cache_data.clear()

    return redirect(url_for('screening'))


@app.route('/screening')
def screening():
    profil_risiko = session.get('profil_risiko', 'Sedang')
    sektor_terpilih = session.get('sektor_terpilih', [])

    if sektor_terpilih:
        tickers = get_saham_dari_sektor(sektor_terpilih)
    else:
        tickers = get_semua_saham()

    try:
        data_harga = _get_data_harga(tickers)

        if data_harga.empty:
            return render_template(
                'error.html',
                pesan='Gagal mengambil data saham. Cek koneksi internet atau coba sektor lain.'
            )

        return_harian = hitung_return_harian(data_harga)
        hasil_rbs = jalankan_rbs(return_harian, profil_risiko)

        saham_lolos = [s['ticker'] for s in hasil_rbs['lolos']]
        meta = _metadata_data(data_harga)

        session['saham_lolos'] = saham_lolos
        session['hasil_rbs_json'] = json.dumps({
            'jumlah_dianalisis': hasil_rbs['jumlah_dianalisis'],
            'jumlah_lolos': hasil_rbs['jumlah_lolos'],
            'jumlah_tidak_lolos': hasil_rbs['jumlah_tidak_lolos'],
            'lolos': hasil_rbs['lolos'],
            'tidak_lolos': hasil_rbs['tidak_lolos'],
        })

        return render_template(
            'screening.html',
            hasil_rbs=hasil_rbs,
            profil_risiko=profil_risiko,
            sektor_terpilih=sektor_terpilih,
            tahun_data=TAHUN_DATA,
            data_mulai=meta['data_mulai'],
            data_selesai=meta['data_selesai'],
        )

    except Exception as exc:
        traceback.print_exc()
        return render_template(
            'error.html',
            pesan=f'Error screening: {exc}'
        )


@app.route('/optimasi')
def optimasi():
    dana_investasi = session.get('dana_investasi', 2_000_000)
    profil_risiko = session.get('profil_risiko', 'Sedang')
    saham_lolos = session.get('saham_lolos', [])

    if len(saham_lolos) < MIN_SAHAM_LOLOS:
        return render_template(
            'error.html',
            pesan=(
                f'Saham yang lolos hanya {len(saham_lolos)}. '
                f'Minimal {MIN_SAHAM_LOLOS}. '
                f'Coba ubah profil risiko ke Tinggi atau pilih lebih banyak sektor.'
            )
        )

    try:
        data_harga = _get_data_harga(saham_lolos)

        if data_harga.empty:
            return render_template(
                'error.html',
                pesan='Data saham lolos tidak tersedia untuk optimasi.'
            )

        rolling = ambil_window_terakhir(data_harga, WINDOW_BULAN)

        mean_return = rolling['mean_return']
        cov_matrix = rolling['cov_matrix']

        min_b, max_b = _batas_bobot(profil_risiko)

        hasil_qp = optimasi_portofolio_qp(
            mean_return,
            cov_matrix,
            min_bobot=min_b,
            max_bobot=max_b
        )

        if hasil_qp.get('status') != 'optimal':
            return render_template(
                'error.html',
                pesan=f"Optimasi gagal: {hasil_qp.get('pesan', '-')}"
            )

        hasil_tampil = format_hasil_untuk_tampilan(
            hasil_qp,
            dana_investasi,
            INFO_SAHAM
        )

        frontier = hitung_efficient_frontier(
            mean_return,
            cov_matrix,
            jumlah_titik=25,
            min_bobot=min_b,
            max_bobot=max_b
        )

        portofolio_titik = {
            'return': round(hasil_qp['expected_return'] * 100, 2),
            'risiko': round(hasil_qp['expected_risk'] * 100, 2),
        }

        meta = _metadata_data(data_harga)

        rbs_json = session.get('hasil_rbs_json')
        hasil_rbs = json.loads(rbs_json) if rbs_json else {}

        session['hasil_optimasi_json'] = json.dumps({
            'alokasi': hasil_tampil['alokasi'],
            'expected_return_pct': hasil_tampil['expected_return_pct'],
            'expected_risk_pct': hasil_tampil['expected_risk_pct'],
            'sharpe_ratio': hasil_tampil['sharpe_ratio'],
            'dana_fmt': hasil_tampil['dana_fmt'],
            'frontier': frontier,
            'portofolio_titik': portofolio_titik,

            # Metadata agar UI tidak membingungkan 5 tahun dan 12 bulan.
            'tahun_data': TAHUN_DATA,
            'data_mulai': meta['data_mulai'],
            'data_selesai': meta['data_selesai'],

            # Window optimasi terbaru.
            'window_mulai': str(rolling['periode_mulai'].date()),
            'window_selesai': str(rolling['periode_selesai'].date()),
            'window_bulan': WINDOW_BULAN,
            'step_bulan': STEP_BULAN,

            'jumlah_saham_valid': meta['jumlah_saham_valid'],
            'jumlah_lolos': len(saham_lolos),
        })

        return render_template(
            'optimasi.html',
            hasil=hasil_tampil,
            frontier_titik=json.dumps(frontier),
            portofolio_titik=json.dumps(portofolio_titik),
            periode_mulai=str(rolling['periode_mulai'].date()),
            periode_selesai=str(rolling['periode_selesai'].date()),
            tahun_data=TAHUN_DATA,
            data_mulai=meta['data_mulai'],
            data_selesai=meta['data_selesai'],
            window_bulan=WINDOW_BULAN,
            jumlah_lolos=len(saham_lolos),
            hasil_rbs=hasil_rbs,
        )

    except Exception as exc:
        traceback.print_exc()
        return render_template(
            'error.html',
            pesan=f'Error optimasi: {exc}'
        )


@app.route('/backtest')
def backtest():
    profil_risiko = session.get('profil_risiko', 'Sedang')
    dana_investasi = session.get('dana_investasi', 2_000_000)
    saham_lolos = session.get('saham_lolos', [])

    if not saham_lolos:
        return redirect(url_for('index'))

    try:
        data_harga = _get_data_harga(saham_lolos)

        if data_harga.empty:
            return render_template(
                'error.html',
                pesan='Data saham tidak tersedia untuk backtesting.'
            )

        hasil_bt = jalankan_backtest(
            data_harga,
            profil_risiko=profil_risiko,
            modal_awal=dana_investasi,
            window_bulan=WINDOW_BULAN,
            step_bulan=STEP_BULAN,
        )

        if hasil_bt.get('status') != 'berhasil':
            return render_template(
                'error.html',
                pesan=hasil_bt.get('pesan', 'Backtesting gagal.')
            )

        chart_data = format_backtest_untuk_chart(hasil_bt)

        session['backtest_json'] = json.dumps({
            'modal_awal_fmt': hasil_bt['modal_awal_fmt'],
            'modal_akhir_fmt': hasil_bt['modal_akhir_fmt'],
            'total_return_pct': hasil_bt['total_return_pct'],
            'return_tahunan_pct': hasil_bt['return_tahunan_pct'],
            'max_drawdown_pct': hasil_bt['max_drawdown_pct'],
            'sharpe_backtest': hasil_bt['sharpe_backtest'],
            'jumlah_window': hasil_bt['jumlah_window'],
            'chart': chart_data,

            # Jadwal rebalancing.
            'jadwal_rebalancing': hasil_bt.get('jadwal_rebalancing', []),
            'rebalancing_pertama': hasil_bt.get('rebalancing_pertama', '-'),
            'rebalancing_terakhir': hasil_bt.get('rebalancing_terakhir', '-'),

            # Metadata periode backtest.
            'data_mulai': hasil_bt['data_mulai'],
            'data_selesai': hasil_bt['data_selesai'],
            'backtest_mulai': hasil_bt['backtest_mulai'],
            'backtest_selesai': hasil_bt['backtest_selesai'],

            # Konfigurasi.
            'window_bulan': hasil_bt['window_bulan'],
            'step_bulan': hasil_bt['step_bulan'],
            'window_hari': hasil_bt.get('window_hari', 252),
            'step_hari': hasil_bt.get('step_hari', 21),
            'tahun_data': TAHUN_DATA,
        })

        return render_template(
            'backtest.html',
            hasil=hasil_bt,
            chart_data=json.dumps(chart_data)
        )

    except Exception as exc:
        traceback.print_exc()
        return render_template(
            'error.html',
            pesan=f'Error backtest: {exc}'
        )


@app.route('/hasil')
def hasil():
    profil_risiko = session.get('profil_risiko', 'Sedang')

    opt_json = session.get('hasil_optimasi_json')
    bt_json = session.get('backtest_json')
    rbs_json = session.get('hasil_rbs_json')

    if not opt_json:
        return redirect(url_for('index'))

    opt = json.loads(opt_json)
    bt = json.loads(bt_json) if bt_json else None
    rbs = json.loads(rbs_json) if rbs_json else {}

    kategori = _kategori_risiko(opt['expected_risk_pct'])

    return render_template(
        'hasil.html',

        # Hasil optimasi.
        alokasi=opt['alokasi'],
        expected_return=opt['expected_return_pct'],
        expected_risk=opt['expected_risk_pct'],
        sharpe_ratio=opt['sharpe_ratio'],
        dana_fmt=opt['dana_fmt'],
        kategori_risiko=kategori,
        profil_risiko=profil_risiko,
        frontier=json.dumps(opt.get('frontier', [])),
        portofolio_titik=json.dumps(opt.get('portofolio_titik', {})),

        # Hasil backtest.
        backtest=bt,
        hasil_rbs=rbs,

        # Metadata periode sistem.
        tahun_data=opt.get('tahun_data', TAHUN_DATA),
        data_mulai=opt.get('data_mulai', '-'),
        data_selesai=opt.get('data_selesai', '-'),
        window_mulai=opt.get('window_mulai', '-'),
        window_selesai=opt.get('window_selesai', '-'),
        window_bulan=opt.get('window_bulan', WINDOW_BULAN),
        step_bulan=opt.get('step_bulan', STEP_BULAN),
        jumlah_saham_valid=opt.get('jumlah_saham_valid', 0),
        jumlah_lolos=opt.get('jumlah_lolos', 0),
    )


@app.route('/reset')
def reset():
    session.clear()
    _cache_data.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    print('=' * 60)
    print(' PortoCUAN - Sistem Optimisasi Portofolio Investor Retail')
    print(' Buka browser: http://localhost:5000')
    print('=' * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
