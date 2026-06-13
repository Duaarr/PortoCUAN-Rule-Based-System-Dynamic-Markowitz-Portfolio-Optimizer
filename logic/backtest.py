"""
backtest.py

Simulasi historis portofolio dengan rolling window dan rebalancing bulanan.

Versi terbaru:
- Data historis tetap memakai data harian.
- Window training = 12 bulan atau ±252 hari perdagangan.
- Rebalancing = setiap 1 bulan atau ±21 hari perdagangan.
- Tanggal rebalancing dibuat jelas:
  hari perdagangan pertama setelah window training selesai.
- Periode uji dibuat jelas:
  21 hari perdagangan setelah tanggal rebalancing.
- Hasil backtesting menjadi lebih mudah dijelaskan karena:
  metadata data historis, window training, tanggal rebalancing,
  dan periode uji dipisahkan.
"""

import numpy as np
import pandas as pd

from logic.markowitz import optimasi_portofolio_qp


def jalankan_backtest(
    data_harga: pd.DataFrame,
    profil_risiko: str = 'Sedang',
    modal_awal: float = 2_000_000,
    window_bulan: int = 12,
    step_bulan: int = 1,
) -> dict:
    """
    Menjalankan backtesting out-of-sample.

    Konsep utama:
    1. Sistem mengambil data training selama 12 bulan perdagangan.
    2. Dari data training tersebut, sistem menghitung:
       - return harian,
       - mean return tahunan,
       - covariance matrix tahunan,
       - bobot optimal Markowitz.
    3. Rebalancing dilakukan pada hari perdagangan pertama
       setelah window training selesai.
    4. Bobot hasil Markowitz diterapkan pada periode uji 1 bulan berikutnya.
    5. Modal diperbarui berdasarkan return portofolio selama periode uji.
    6. Window digeser 1 bulan perdagangan, lalu proses diulang.
    """

    if data_harga is None or data_harga.empty:
        return {
            'status': 'gagal',
            'pesan': 'Data harga kosong.'
        }

    print(f"[Backtest] Mulai | Modal Rp {modal_awal:,.0f} | Profil {profil_risiko}")

    batas_bobot = {
        'Rendah': (0.05, 0.35),
        'Sedang': (0.05, 0.40),
        'Tinggi': (0.05, 0.50),
    }

    min_bobot, max_bobot = batas_bobot.get(profil_risiko, (0.05, 0.40))

    # Dalam implementasi ini, 1 bulan dianggap sekitar 21 hari perdagangan.
    hari_per_bulan = 21
    window_hari = window_bulan * hari_per_bulan
    step_hari = step_bulan * hari_per_bulan

    jumlah_minimal_hari = window_hari + step_hari

    if len(data_harga) < jumlah_minimal_hari:
        return {
            'status': 'gagal',
            'pesan': (
                f'Data tidak cukup untuk backtesting. '
                f'Dibutuhkan minimal {jumlah_minimal_hari} hari perdagangan, '
                f'tetapi data tersedia hanya {len(data_harga)} hari.'
            )
        }

    riwayat_modal = []
    jadwal_rebalancing = []
    return_list = []

    modal_sekarang = float(modal_awal)

    posisi = 0
    periode_ke = 1

    while posisi + window_hari + step_hari <= len(data_harga):

        # 1. Ambil data training 12 bulan
        data_training = data_harga.iloc[
            posisi: posisi + window_hari
        ].copy()

        # 2. Ambil data testing 1 bulan setelah training
        data_testing = data_harga.iloc[
            posisi + window_hari: posisi + window_hari + step_hari
        ].copy()

        if data_training.empty or data_testing.empty:
            posisi += step_hari
            continue

        # 3. Hitung return harian dari data training
        return_training = (
            data_training
            .pct_change()
            .dropna(how='all')
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        if return_training.empty or return_training.shape[1] < 2:
            posisi += step_hari
            continue

        # 4. Hitung mean return dan covariance matrix tahunan
        mean_ret = return_training.mean() * 252
        cov_mat = return_training.cov() * 252

        # Stabilkan covariance matrix agar aman untuk optimasi QP.
        arr = cov_mat.values.astype(float)
        arr = (arr + arr.T) / 2
        arr = arr + np.eye(len(mean_ret)) * 1e-8

        cov_fix = pd.DataFrame(
            arr,
            index=cov_mat.index,
            columns=cov_mat.columns
        )

        # 5. Pilih saham valid
        valid = mean_ret.dropna().index.tolist()

        if len(valid) < 2:
            posisi += step_hari
            continue

        # Pada backtest, saham yang memiliki mean return positif lebih diprioritaskan.
        valid_positif = mean_ret[mean_ret > 0].dropna().index.tolist()

        if len(valid_positif) >= 2:
            valid = valid_positif

        # 6. Jalankan optimasi Markowitz
        hasil_qp = optimasi_portofolio_qp(
            mean_ret[valid],
            cov_fix.loc[valid, valid],
            min_bobot=min_bobot,
            max_bobot=max_bobot
        )

        if hasil_qp.get('status') == 'optimal':
            bobot = hasil_qp['bobot']
        else:
            # Fallback kalau optimasi gagal.
            bobot = {saham: 1.0 / len(valid) for saham in valid}

        # 7. Terapkan bobot ke data testing
        return_testing = (
            data_testing
            .pct_change()
            .dropna(how='all')
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        tersedia = [saham for saham in bobot if saham in return_testing.columns]

        if len(tersedia) < 2:
            posisi += step_hari
            continue

        bobot_tersedia = {saham: bobot[saham] for saham in tersedia}
        total_bobot = sum(bobot_tersedia.values())

        if total_bobot <= 0:
            posisi += step_hari
            continue

        # Normalisasi ulang bobot untuk saham yang tersedia pada periode testing.
        bobot_normal = {
            saham: nilai / total_bobot
            for saham, nilai in bobot_tersedia.items()
        }

        bobot_arr = np.array([bobot_normal[saham] for saham in tersedia])

        # Return portofolio harian = return saham harian × bobot saham.
        ret_portofolio_harian = return_testing[tersedia].values @ bobot_arr

        # Compound return selama periode testing.
        return_aktual = float(np.prod(1 + ret_portofolio_harian) - 1)

        # Guard agar return ekstrem tidak merusak grafik.
        return_aktual = max(-0.80, min(1.00, return_aktual))

        modal_sekarang *= (1 + return_aktual)
        return_list.append(return_aktual)

        # 8. Tentukan tanggal penting
        tanggal_training_mulai = data_training.index[0]
        tanggal_training_selesai = data_training.index[-1]

        # Tanggal rebalancing = hari pertama periode testing.
        tanggal_rebalancing = data_testing.index[0]

        tanggal_testing_mulai = data_testing.index[0]
        tanggal_testing_selesai = data_testing.index[-1]

        # 9. Simpan riwayat modal untuk grafik
        riwayat_modal.append({
            'periode': periode_ke,

            'tanggal': str(tanggal_testing_selesai.date()),
            'tanggal_label': tanggal_testing_selesai.strftime('%b %Y'),

            'nilai_modal': round(modal_sekarang, 0),
            'return_bulan': round(return_aktual * 100, 2),

            'training_mulai': str(tanggal_training_mulai.date()),
            'training_selesai': str(tanggal_training_selesai.date()),
            'tanggal_rebalancing': str(tanggal_rebalancing.date()),
            'testing_mulai': str(tanggal_testing_mulai.date()),
            'testing_selesai': str(tanggal_testing_selesai.date()),
        })

        # 10. Simpan jadwal rebalancing untuk tabel UI
        jadwal_rebalancing.append({
            'periode': periode_ke,

            'training_mulai': str(tanggal_training_mulai.date()),
            'training_selesai': str(tanggal_training_selesai.date()),

            'tanggal_rebalancing': str(tanggal_rebalancing.date()),

            'testing_mulai': str(tanggal_testing_mulai.date()),
            'testing_selesai': str(tanggal_testing_selesai.date()),

            'return_periode': round(return_aktual * 100, 2),
            'modal_akhir': round(modal_sekarang, 0),

            'jumlah_saham': len(tersedia),
        })

        posisi += step_hari
        periode_ke += 1

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
        'status': 'berhasil',

        # Riwayat utama.
        'riwayat_modal': riwayat_modal,
        'jadwal_rebalancing': jadwal_rebalancing,

        # Modal.
        'modal_awal': modal_awal,
        'modal_awal_fmt': f"Rp {modal_awal:,.0f}".replace(',', '.'),

        'modal_akhir': round(modal_akhir, 0),
        'modal_akhir_fmt': f"Rp {modal_akhir:,.0f}".replace(',', '.'),

        # Metrik backtest.
        'total_return_pct': round(total_return_pct, 2),
        'return_tahunan_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_backtest': round(sharpe, 4),
        'jumlah_window': len(riwayat_modal),

        # Metadata data historis.
        'data_mulai': str(data_harga.index[0].date()),
        'data_selesai': str(data_harga.index[-1].date()),

        # Periode backtest yang benar-benar diuji.
        'backtest_mulai': riwayat_modal[0]['testing_mulai'],
        'backtest_selesai': riwayat_modal[-1]['testing_selesai'],

        # Tanggal rebalancing.
        'rebalancing_pertama': jadwal_rebalancing[0]['tanggal_rebalancing'],
        'rebalancing_terakhir': jadwal_rebalancing[-1]['tanggal_rebalancing'],

        # Konfigurasi.
        'window_bulan': window_bulan,
        'step_bulan': step_bulan,
        'window_hari': window_hari,
        'step_hari': step_hari,
    }


def _max_drawdown(riwayat: list[dict]) -> float:
    """
    Menghitung penurunan terbesar dari puncak modal ke lembah berikutnya.
    """

    nilai = [x['nilai_modal'] for x in riwayat]

    if not nilai:
        return 0.0

    puncak = nilai[0]
    max_dd = 0.0

    for v in nilai:
        puncak = max(puncak, v)
        dd = (v - puncak) / puncak * 100
        max_dd = min(max_dd, dd)

    return max_dd


def _sharpe_backtest(return_list: list[float]) -> float:
    """
    Menghitung Sharpe Ratio dari return periode backtesting.

    Karena return_list berisi return bulanan/periode,
    maka:
    - mean return dikalikan 12,
    - standar deviasi dikalikan akar 12.
    """

    if len(return_list) < 2:
        return 0.0

    r = np.array(return_list, dtype=float)

    mean_annual = np.mean(r) * 12
    std_annual = np.std(r) * np.sqrt(12)

    if std_annual <= 0:
        return 0.0

    risk_free_rate = 0.06

    return float((mean_annual - risk_free_rate) / std_annual)


def format_backtest_untuk_chart(hasil: dict) -> dict:
    """
    Format data agar bisa langsung dipakai Chart.js.

    Label grafik memakai akhir periode uji, bukan tanggal metadata.
    Tanggal rebalancing detail ditampilkan di tabel jadwal rebalancing.
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
