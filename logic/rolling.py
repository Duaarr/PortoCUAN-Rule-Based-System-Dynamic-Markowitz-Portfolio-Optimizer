"""
rolling.py
Membuat rolling window untuk model Markowitz dinamis.

Makna penting:
- Data historis bisa 5 tahun.
- Window Markowitz default 12 bulan.
- Backtest menggeser window setiap 1 bulan.
"""

import pandas as pd


def _bersihkan_return(return_harian: pd.DataFrame, min_kolom: int = 2) -> pd.DataFrame:
    """
    Membersihkan return supaya kovarian stabil.
    """
    if return_harian is None or return_harian.empty:
        return pd.DataFrame()

    rh = return_harian.dropna(axis=1, how='all')

    if rh.empty:
        return pd.DataFrame()

    thresh = int(len(rh) * 0.90)
    rh_bersih = rh.dropna(axis=1, thresh=thresh).fillna(0)

    if rh_bersih.shape[1] < min_kolom:
        rh_bersih = rh.fillna(0)

    return rh_bersih


def rolling_return_dan_kovarian(data_harga: pd.DataFrame, window_bulan: int = 12) -> dict:
    """
    Mengambil window terakhir N bulan untuk rekomendasi portofolio saat ini.
    """
    if data_harga is None or data_harga.empty:
        raise ValueError('Data harga kosong.')

    hari_per_bulan = 21
    window_hari = window_bulan * hari_per_bulan

    if len(data_harga) < window_hari:
        raise ValueError(f'Data tidak cukup untuk window {window_bulan} bulan.')

    data_window = data_harga.iloc[-window_hari:].copy()

    return_raw = data_window.pct_change().dropna(how='all')
    return_harian = _bersihkan_return(return_raw)

    if return_harian.empty or return_harian.shape[1] < 2:
        raise ValueError('Return harian tidak cukup untuk optimasi.')

    mean_return = return_harian.mean() * 252
    cov_matrix = return_harian.cov() * 252

    return {
        'mean_return': mean_return,
        'cov_matrix': cov_matrix,
        'return_harian': return_harian,
        'periode_mulai': data_window.index[0],
        'periode_selesai': data_window.index[-1],
        'jumlah_saham': return_harian.shape[1],
        'jumlah_hari': len(return_harian),
    }


def ambil_window_terakhir(data_harga: pd.DataFrame, window_bulan: int = 12) -> dict:
    """
    Alias agar lebih mudah dibaca di app.py.
    """
    return rolling_return_dan_kovarian(data_harga, window_bulan)


def simulasi_semua_window(
    data_harga: pd.DataFrame,
    window_bulan: int = 12,
    step_bulan: int = 1
) -> list[dict]:
    """
    Membuat semua rolling window untuk backtesting.
    Contoh:
    - window 12 bulan,
    - step 1 bulan,
    - maka model dihitung ulang setiap bulan.
    """
    if data_harga is None or data_harga.empty:
        return []

    hari_per_bulan = 21
    window_hari = window_bulan * hari_per_bulan
    step_hari = step_bulan * hari_per_bulan

    hasil = []

    if len(data_harga) < window_hari + step_hari:
        return hasil

    posisi = 0

    while posisi + window_hari <= len(data_harga):
        data_window = data_harga.iloc[posisi:posisi + window_hari].copy()

        try:
            return_raw = data_window.pct_change().dropna(how='all')
            return_harian = _bersihkan_return(return_raw)

            if return_harian.empty or return_harian.shape[1] < 2:
                posisi += step_hari
                continue

            mean_return = return_harian.mean() * 252
            cov_matrix = return_harian.cov() * 252

            hasil.append({
                'mean_return': mean_return,
                'cov_matrix': cov_matrix,
                'return_harian': return_harian,
                'periode_mulai': data_window.index[0],
                'periode_selesai': data_window.index[-1],
                'jumlah_saham': return_harian.shape[1],
            })

        except Exception as exc:
            print(f"[Rolling] Window dilewati: {exc}")

        posisi += step_hari

    print(f"[Rolling] Total window dibuat: {len(hasil)}")

    return hasil