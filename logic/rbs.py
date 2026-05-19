"""
rbs.py
Rule-Based System untuk menyaring saham sebelum masuk ke Markowitz.

Aturan umum:
- return tahunan harus memenuhi batas minimum,
- volatilitas tahunan tidak boleh melebihi batas profil risiko,
- Sharpe Ratio harus memenuhi nilai minimum.
"""

import numpy as np
import pandas as pd


PROFIL_RISIKO_CONFIG = {
    'Rendah': {
        'min_return_tahunan': 0.05,
        'max_volatilitas': 0.20,
        'min_sharpe': 0.20,
        'label': 'Konservatif'
    },
    'Sedang': {
        'min_return_tahunan': 0.08,
        'max_volatilitas': 0.30,
        'min_sharpe': 0.10,
        'label': 'Moderat'
    },
    'Tinggi': {
        'min_return_tahunan': 0.10,
        'max_volatilitas': 0.50,
        'min_sharpe': 0.00,
        'label': 'Agresif'
    }
}

RISK_FREE_RATE_TAHUNAN = 0.06


def hitung_statistik_saham(return_harian: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung return, volatilitas, dan Sharpe Ratio tahunan.
    """
    if return_harian is None or return_harian.empty:
        return pd.DataFrame(
            columns=['return_tahunan', 'volatilitas_tahunan', 'sharpe_ratio']
        )

    mean_return_harian = return_harian.mean()
    return_tahunan = mean_return_harian * 252

    volatilitas_tahunan = return_harian.std() * np.sqrt(252)

    sharpe_ratio = (
        return_tahunan - RISK_FREE_RATE_TAHUNAN
    ) / volatilitas_tahunan.replace(0, np.nan)

    statistik = pd.DataFrame({
        'return_tahunan': return_tahunan,
        'volatilitas_tahunan': volatilitas_tahunan,
        'sharpe_ratio': sharpe_ratio
    })

    statistik = statistik.replace([np.inf, -np.inf], np.nan).dropna()

    return statistik.round(4)


def jalankan_rbs(return_harian: pd.DataFrame, profil_risiko: str) -> dict:
    """
    Menyaring saham lolos dan tidak lolos berdasarkan profil risiko.
    """
    config = PROFIL_RISIKO_CONFIG.get(
        profil_risiko,
        PROFIL_RISIKO_CONFIG['Sedang']
    )

    statistik = hitung_statistik_saham(return_harian)

    lolos = []
    tidak_lolos = []

    for ticker in statistik.index:
        ret = float(statistik.loc[ticker, 'return_tahunan'])
        vol = float(statistik.loc[ticker, 'volatilitas_tahunan'])
        shrp = float(statistik.loc[ticker, 'sharpe_ratio'])

        nama = ticker.replace('.JK', '')
        sektor = _get_sektor(ticker)

        alasan_gagal = None

        if ret < config['min_return_tahunan']:
            alasan_gagal = (
                f"Return rendah ({ret * 100:.1f}% < "
                f"{config['min_return_tahunan'] * 100:.0f}%)"
            )

        elif vol > config['max_volatilitas']:
            alasan_gagal = (
                f"Volatilitas melebihi batas ({vol * 100:.1f}% > "
                f"{config['max_volatilitas'] * 100:.0f}%)"
            )

        elif shrp < config['min_sharpe']:
            alasan_gagal = (
                f"Sharpe Ratio rendah ({shrp:.2f} < "
                f"{config['min_sharpe']:.2f})"
            )

        if alasan_gagal:
            tidak_lolos.append(
                _buat_entry(nama, ticker, sektor, ret, vol, shrp, alasan_gagal)
            )
        else:
            lolos.append(
                _buat_entry(
                    nama,
                    ticker,
                    sektor,
                    ret,
                    vol,
                    shrp,
                    _alasan_lolos(ret, vol, profil_risiko)
                )
            )

    lolos.sort(key=lambda x: (x['sharpe'], x['return_pct']), reverse=True)
    tidak_lolos.sort(key=lambda x: x['saham'])

    return {
        'lolos': lolos,
        'tidak_lolos': tidak_lolos,
        'statistik': statistik,
        'config': config,
        'profil_risiko': profil_risiko,
        'jumlah_lolos': len(lolos),
        'jumlah_tidak_lolos': len(tidak_lolos),
        'jumlah_dianalisis': len(statistik)
    }


def _buat_entry(nama, ticker, sektor, ret, vol, sharpe, alasan) -> dict:
    return {
        'saham': nama,
        'ticker': ticker,
        'sektor': sektor,
        'return_pct': round(ret * 100, 2),
        'volatilitas_pct': round(vol * 100, 2),
        'sharpe': round(sharpe, 2),
        'alasan': alasan
    }


def _get_sektor(ticker: str) -> str:
    try:
        from logic.markowitz import INFO_SAHAM
        return INFO_SAHAM.get(ticker, {}).get('sektor', '-')
    except Exception:
        return '-'


def _alasan_lolos(ret: float, vol: float, profil: str) -> str:
    if ret >= 0.20 and vol < 0.20:
        return 'Return tinggi dengan volatilitas relatif rendah'

    if vol < 0.15:
        return 'Volatilitas terkendali dan memenuhi profil risiko'

    if profil == 'Tinggi' and ret >= 0.10:
        return 'Return memenuhi profil agresif'

    return 'Memenuhi batas return, risiko, dan Sharpe Ratio'