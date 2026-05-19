"""
data_fetcher.py
Mengambil dan membersihkan data harga saham dari Yahoo Finance.

Fungsi utama:
- get_semua_saham()
- get_saham_dari_sektor()
- ambil_data_saham()
- hitung_return_harian()
"""

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


SEKTOR_SAHAM = {
    'Perbankan': [
        'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BRIS.JK',
        'BTPS.JK', 'ARTO.JK', 'BDMN.JK', 'PNBN.JK', 'BJBR.JK'
    ],
    'Properti': [
        'PWON.JK', 'BKSL.JK', 'PANI.JK', 'CTRA.JK', 'BSDE.JK',
        'SMRA.JK', 'ASRI.JK', 'DILD.JK', 'MKPI.JK', 'JRPT.JK'
    ],
    'Infrastruktur': [
        'TLKM.JK', 'JSMR.JK', 'PGAS.JK', 'WIKA.JK', 'PTPP.JK',
        'ADHI.JK', 'EXCL.JK', 'ISAT.JK', 'TOWR.JK', 'TBIG.JK'
    ],
    'Konsumen': [
        'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'MYOR.JK', 'GGRM.JK',
        'HMSP.JK', 'KLBF.JK', 'AMRT.JK', 'CPIN.JK', 'JPFA.JK'
    ],
    'Energi': [
        'ADRO.JK', 'PTBA.JK', 'ITMG.JK', 'HRUM.JK', 'MEDC.JK',
        'AKRA.JK', 'UNTR.JK', 'INDY.JK', 'MBMA.JK', 'ENRG.JK'
    ]
}


def get_semua_saham() -> list[str]:
    """Menggabungkan semua ticker dari seluruh sektor."""
    semua = []

    for daftar in SEKTOR_SAHAM.values():
        semua.extend(daftar)

    return semua


def get_saham_dari_sektor(sektor_terpilih: list[str]) -> list[str]:
    """Mengambil ticker berdasarkan sektor yang dipilih user."""
    hasil = []

    for sektor in sektor_terpilih:
        hasil.extend(SEKTOR_SAHAM.get(sektor, []))

    return hasil


def _ekstrak_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Mengambil harga Close dari hasil yfinance.
    Dibuat aman untuk format yfinance lama dan baru.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        if 'Close' in raw.columns:
            data = raw[['Close']].copy()
            data.columns = [tickers[0] if tickers else 'saham']
            return data

        return pd.DataFrame()

    level0 = raw.columns.get_level_values(0).unique().tolist()
    level1 = raw.columns.get_level_values(1).unique().tolist()

    if 'Close' in level0:
        return raw['Close'].copy()

    if 'Close' in level1:
        return raw.xs('Close', axis=1, level=1).copy()

    return pd.DataFrame()


def ambil_data_saham(tickers: list[str], tahun: int = 5) -> pd.DataFrame:
    """
    Mengambil harga penutupan saham selama N tahun terakhir.
    Periode ini adalah data historis utama, bukan window Markowitz.
    """
    if not tickers:
        return pd.DataFrame()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=tahun * 365)

    print(f"[DATA] Mengambil {len(tickers)} saham | {start_date.date()} → {end_date.date()}")

    try:
        raw = yf.download(
            tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        data = _ekstrak_close(raw, tickers)

        if data.empty:
            print('[DATA] Gagal mengambil kolom Close.')
            return pd.DataFrame()

        data.columns = [str(c) for c in data.columns]

        # Buang saham yang semua datanya kosong.
        data = data.dropna(axis=1, how='all')

        # Isi data kosong dengan harga sebelumnya.
        data = data.ffill().dropna()

        if data.empty:
            print('[DATA] Data kosong setelah pembersihan.')
            return pd.DataFrame()

        print(f"[DATA] OK — {data.shape[1]} saham valid, {data.shape[0]} hari perdagangan")

        return data

    except Exception as exc:
        print(f"[DATA] Error: {exc}")
        return pd.DataFrame()


def hitung_return_harian(data: pd.DataFrame) -> pd.DataFrame:
    """Menghitung return harian dari harga saham."""
    if data is None or data.empty:
        return pd.DataFrame()

    return data.pct_change().dropna(how='all').fillna(0)


def hitung_return_bulanan(data: pd.DataFrame) -> pd.DataFrame:
    """Menghitung return bulanan berdasarkan harga akhir bulan."""
    if data is None or data.empty:
        return pd.DataFrame()

    data_bulanan = data.resample('ME').last()

    return data_bulanan.pct_change().dropna(how='all').fillna(0)