"""
backtest.py
Simulasi historis portofolio dengan rolling window dan rebalancing.

Alur:
1. Ambil window 12 bulan.
2. Hitung bobot Markowitz dari window tersebut.
3. Terapkan bobot ke bulan berikutnya.
4. Update modal.
5. Ulangi sampai akhir data.
"""

import numpy as np
import pandas as pd
from logic.markowitz import optimasi_portofolio_qp
from logic.rolling import simulasi_semua_window


def jalankan_backtest(
    data_harga: pd.DataFrame,
    profil_risiko: str = 'Sedang',
    modal_awal: float = 2_000_000,
    window_bulan: int = 12,
    step_bulan: int = 1,
) -> dict:
    """
    Menjalankan backtesting out-of-sample.
    """
    if data_harga is None or data_harga.empty:
        return {'status': 'gagal', 'pesan': 'Data harga kosong.'}

    print(f"[Backtest] Mulai | Modal Rp {modal_awal:,.0f} | Profil {profil_risiko}")

    batas = {
        'Rendah': (0.05, 0.35),
        'Sedang': (0.05, 0.40),
        'Tinggi': (0.05, 0.50)
    }

    min_b, max_b = batas.get(profil_risiko, (0.05, 0.40))

    semua_window = simulasi_semua_window(
        data_harga,
        window_bulan,
        step_bulan
    )

    if len(semua_window) < 2:
        return {
            'status': 'gagal',
            'pesan': f'Data tidak cukup untuk backtest. Tersedia {len(semua_window)} window, minimal 2.'
        }

    riwayat_modal = []
    modal_sekarang = float(modal_awal)
    return_list = []

    hari_step = step_bulan * 21

    for i, window in enumerate(semua_window[:-1]):
        mean_ret = window['mean_return']
        cov_mat = window['cov_matrix']

        arr = cov_mat.values.astype(float)
        arr = (arr + arr.T) / 2 + np.eye(len(mean_ret)) * 1e-8
        cov_fix = pd.DataFrame(
            arr,
            index=cov_mat.index,
            columns=cov_mat.columns
        )

        # Saham valid untuk window ini.
        valid = mean_ret[mean_ret > 0].index.tolist()

        if len(valid) < 2:
            valid = mean_ret.dropna().index.tolist()

        if len(valid) < 2:
            continue

        hasil_qp = optimasi_portofolio_qp(
            mean_ret[valid],
            cov_fix.loc[valid, valid],
            min_bobot=min_b,
            max_bobot=max_b
        )

        if hasil_qp.get('status') == 'optimal':
            bobot = hasil_qp['bobot']
        else:
            # Fallback kalau optimasi gagal.
            bobot = {s: 1.0 / len(valid) for s in valid}

        # Terapkan bobot ke window berikutnya.
        ret_berikut = semua_window[i + 1]['return_harian']

        tersedia = [s for s in bobot if s in ret_berikut.columns]

        if not tersedia:
            continue

        bobot_t = {s: bobot[s] for s in tersedia}
        total_b = sum(bobot_t.values())

        if total_b <= 0:
            continue

        bobot_n = {s: v / total_b for s, v in bobot_t.items()}

        ret_periode = ret_berikut[tersedia].iloc[:hari_step].fillna(0)

        if ret_periode.empty:
            continue

        bobot_arr = np.array([bobot_n[s] for s in tersedia])

        ret_portofolio_harian = ret_periode.values @ bobot_arr

        # Compound return bulanan.
        return_aktual = float(np.prod(1 + ret_portofolio_harian) - 1)

        # Guard agar return ekstrem tidak merusak grafik.
        return_aktual = max(-0.80, min(1.00, return_aktual))

        modal_sekarang *= (1 + return_aktual)

        return_list.append(return_aktual)

        tanggal = semua_window[i + 1]['periode_selesai']

        riwayat_modal.append({
            'tanggal': str(tanggal.date()),
            'tanggal_label': tanggal.strftime('%b %Y'),
            'nilai_modal': round(modal_sekarang, 0),
            'return_bulan': round(return_aktual * 100, 2),
        })

    if not riwayat_modal:
        return {
            'status': 'gagal',
            'pesan': 'Tidak ada periode backtest yang berhasil dihitung.'
        }

    modal_akhir = modal_sekarang

    total_return_pct = (modal_akhir / modal_awal - 1) * 100

    n_tahun = len(riwayat_modal) / 12

    if n_tahun > 0 and modal_akhir > 0:
        cagr = ((modal_akhir / modal_awal) ** (1 / n_tahun) - 1) * 100
    else:
        cagr = 0

    max_dd = _max_drawdown(riwayat_modal)
    sharpe = _sharpe_backtest(return_list)

    return {
        'riwayat_modal': riwayat_modal,

        'modal_awal': modal_awal,
        'modal_awal_fmt': f"Rp {modal_awal:,.0f}".replace(',', '.'),

        'modal_akhir': round(modal_akhir, 0),
        'modal_akhir_fmt': f"Rp {modal_akhir:,.0f}".replace(',', '.'),

        'total_return_pct': round(total_return_pct, 2),
        'return_tahunan_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_backtest': round(sharpe, 4),
        'jumlah_window': len(riwayat_modal),

        # Metadata agar UI jelas.
        'data_mulai': str(data_harga.index[0].date()),
        'data_selesai': str(data_harga.index[-1].date()),
        'backtest_mulai': riwayat_modal[0]['tanggal'],
        'backtest_selesai': riwayat_modal[-1]['tanggal'],
        'window_bulan': window_bulan,
        'step_bulan': step_bulan,

        'status': 'berhasil'
    }


def _max_drawdown(riwayat: list[dict]) -> float:
    """
    Menghitung penurunan terbesar dari puncak ke lembah.
    """
    nilai = [x['nilai_modal'] for x in riwayat]

    puncak = nilai[0]
    max_dd = 0.0

    for v in nilai:
        puncak = max(puncak, v)
        dd = (v - puncak) / puncak * 100
        max_dd = min(max_dd, dd)

    return max_dd


def _sharpe_backtest(return_list: list[float]) -> float:
    """
    Sharpe Ratio dari return bulanan yang dianualisasi.
    """
    if len(return_list) < 2:
        return 0.0

    r = np.array(return_list, dtype=float)

    mean_annual = np.mean(r) * 12
    std_annual = np.std(r) * np.sqrt(12)

    if std_annual <= 0:
        return 0.0

    return float((mean_annual - 0.06) / std_annual)


def format_backtest_untuk_chart(hasil: dict) -> dict:
    """
    Format data agar bisa langsung dipakai Chart.js.
    """
    if hasil.get('status') != 'berhasil':
        return {
            'labels': [],
            'nilai_modal': [],
            'return_bulan': []
        }

    riwayat = hasil['riwayat_modal']

    return {
        'labels': ['Awal'] + [x['tanggal_label'] for x in riwayat],
        'nilai_modal': [hasil['modal_awal']] + [x['nilai_modal'] for x in riwayat],
        'return_bulan': [0] + [x['return_bulan'] for x in riwayat],
    }