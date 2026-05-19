"""
markowitz.py
Model Markowitz dengan Quadratic Programming.

Tujuan:
- menentukan bobot optimal,
- meminimalkan risiko portofolio,
- menjaga total bobot = 100%,
- membatasi bobot minimum dan maksimum sesuai profil risiko.
"""

from typing import Optional
import numpy as np
import pandas as pd
import cvxpy as cp


def optimasi_portofolio_qp(
    mean_return: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: Optional[float] = None,
    min_bobot: float = 0.05,
    max_bobot: float = 0.40
) -> dict:
    """
    Optimasi bobot portofolio menggunakan Quadratic Programming.
    """
    if mean_return is None or cov_matrix is None:
        return {'status': 'gagal', 'pesan': 'Input mean return atau covariance kosong.'}

    if len(mean_return) < 2:
        return {'status': 'gagal', 'pesan': 'Minimal butuh 2 saham untuk optimasi.'}

    saham = list(mean_return.index)
    n = len(saham)

    mu = mean_return.values.astype(float)
    sigma = cov_matrix.loc[saham, saham].values.astype(float)

    # Regularisasi supaya matriks kovarian lebih stabil.
    sigma = (sigma + sigma.T) / 2
    sigma = sigma + np.eye(n) * 1e-8

    w = cp.Variable(n)

    risiko = cp.quad_form(w, sigma)
    objective = cp.Minimize(risiko)

    constraints = [
        cp.sum(w) == 1,
        w >= min_bobot,
        w <= max_bobot
    ]

    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(
            solver=cp.OSQP,
            warm_starting=True,
            eps_abs=1e-6,
            eps_rel=1e-6
        )

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            problem.solve(solver=cp.SCS)

        if problem.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            bobot_array = np.maximum(w.value, 0)
            bobot_array = bobot_array / bobot_array.sum()

            ret = float(mu @ bobot_array)
            risk = float(np.sqrt(bobot_array @ sigma @ bobot_array))
            sharpe = (ret - 0.06) / risk if risk > 0 else 0

            return {
                'bobot': {
                    saham[i]: round(float(bobot_array[i]), 4)
                    for i in range(n)
                },
                'expected_return': round(ret, 4),
                'expected_risk': round(risk, 4),
                'sharpe_ratio': round(sharpe, 4),
                'status': 'optimal'
            }

        return {'status': 'gagal', 'pesan': problem.status}

    except Exception as exc:
        return {'status': 'gagal', 'pesan': str(exc)}


def hitung_efficient_frontier(
    mean_return: pd.Series,
    cov_matrix: pd.DataFrame,
    jumlah_titik: int = 25,
    min_bobot: float = 0.05,
    max_bobot: float = 0.40
) -> list[dict]:
    """
    Membuat titik efficient frontier untuk grafik.
    """
    if mean_return is None or mean_return.empty:
        return []

    ret_min = float(mean_return.min()) * 0.8
    ret_max = float(mean_return.max()) * 0.9

    if ret_min >= ret_max:
        ret_min, ret_max = ret_max * 0.8, ret_min * 1.2

    targets = np.linspace(ret_min, ret_max, jumlah_titik)
    titik = []

    for target in targets:
        hasil = optimasi_portofolio_qp(
            mean_return,
            cov_matrix,
            target_return=target,
            min_bobot=min_bobot,
            max_bobot=max_bobot
        )

        if hasil.get('status') == 'optimal':
            titik.append({
                'return': round(hasil['expected_return'] * 100, 2),
                'risiko': round(hasil['expected_risk'] * 100, 2),
            })

    return titik


def format_hasil_untuk_tampilan(
    hasil_qp: dict,
    dana_investasi: float,
    saham_info: dict | None = None
) -> dict:
    """
    Mengubah hasil optimasi menjadi format yang mudah ditampilkan di HTML.
    """
    if hasil_qp.get('status') != 'optimal':
        return hasil_qp

    alokasi = []

    for ticker, bobot in hasil_qp['bobot'].items():
        rupiah = bobot * dana_investasi
        info = (saham_info or {}).get(ticker, {})

        alokasi.append({
            'ticker': ticker.replace('.JK', ''),
            'ticker_full': ticker,
            'nama': info.get('nama', ticker.replace('.JK', '')),
            'sektor': info.get('sektor', '-'),
            'bobot_pct': round(float(bobot) * 100, 2),
            'rupiah': round(rupiah, 2),
            'rupiah_fmt': f"Rp {rupiah:,.0f}".replace(',', '.')
        })

    alokasi.sort(key=lambda item: item['bobot_pct'], reverse=True)

    return {
        'alokasi': alokasi,
        'expected_return_pct': round(hasil_qp['expected_return'] * 100, 2),
        'expected_risk_pct': round(hasil_qp['expected_risk'] * 100, 2),
        'sharpe_ratio': hasil_qp['sharpe_ratio'],
        'dana_investasi': dana_investasi,
        'dana_fmt': f"Rp {dana_investasi:,.0f}".replace(',', '.'),
        'status': 'optimal'
    }


INFO_SAHAM = {
    'BBCA.JK': {'nama': 'Bank Central Asia', 'sektor': 'Perbankan'},
    'BBRI.JK': {'nama': 'Bank Rakyat Indonesia', 'sektor': 'Perbankan'},
    'BMRI.JK': {'nama': 'Bank Mandiri', 'sektor': 'Perbankan'},
    'BBNI.JK': {'nama': 'Bank Negara Indonesia', 'sektor': 'Perbankan'},
    'BRIS.JK': {'nama': 'Bank Syariah Indonesia', 'sektor': 'Perbankan'},
    'BTPS.JK': {'nama': 'Bank BTPN Syariah', 'sektor': 'Perbankan'},
    'ARTO.JK': {'nama': 'Bank Jago', 'sektor': 'Perbankan'},
    'BDMN.JK': {'nama': 'Bank Danamon', 'sektor': 'Perbankan'},
    'PNBN.JK': {'nama': 'Bank Panin', 'sektor': 'Perbankan'},
    'BJBR.JK': {'nama': 'Bank BJB', 'sektor': 'Perbankan'},

    'PWON.JK': {'nama': 'Pakuwon Jati', 'sektor': 'Properti'},
    'BKSL.JK': {'nama': 'Sentul City', 'sektor': 'Properti'},
    'PANI.JK': {'nama': 'Pantai Indah Kapuk 2', 'sektor': 'Properti'},
    'CTRA.JK': {'nama': 'Ciputra Development', 'sektor': 'Properti'},
    'BSDE.JK': {'nama': 'BSD City', 'sektor': 'Properti'},
    'SMRA.JK': {'nama': 'Summarecon Agung', 'sektor': 'Properti'},
    'ASRI.JK': {'nama': 'Alam Sutera Realty', 'sektor': 'Properti'},
    'DILD.JK': {'nama': 'Intiland Development', 'sektor': 'Properti'},
    'MKPI.JK': {'nama': 'Metropolitan Kentjana', 'sektor': 'Properti'},
    'JRPT.JK': {'nama': 'Jaya Real Property', 'sektor': 'Properti'},

    'TLKM.JK': {'nama': 'Telkom Indonesia', 'sektor': 'Infrastruktur'},
    'JSMR.JK': {'nama': 'Jasa Marga', 'sektor': 'Infrastruktur'},
    'PGAS.JK': {'nama': 'Perusahaan Gas Negara', 'sektor': 'Infrastruktur'},
    'WIKA.JK': {'nama': 'Wijaya Karya', 'sektor': 'Infrastruktur'},
    'PTPP.JK': {'nama': 'PP Persero', 'sektor': 'Infrastruktur'},
    'ADHI.JK': {'nama': 'Adhi Karya', 'sektor': 'Infrastruktur'},
    'EXCL.JK': {'nama': 'XL Axiata', 'sektor': 'Infrastruktur'},
    'ISAT.JK': {'nama': 'Indosat Ooredoo', 'sektor': 'Infrastruktur'},
    'TOWR.JK': {'nama': 'Sarana Menara Nusantara', 'sektor': 'Infrastruktur'},
    'TBIG.JK': {'nama': 'Tower Bersama', 'sektor': 'Infrastruktur'},

    'UNVR.JK': {'nama': 'Unilever Indonesia', 'sektor': 'Konsumen'},
    'ICBP.JK': {'nama': 'Indofood CBP', 'sektor': 'Konsumen'},
    'INDF.JK': {'nama': 'Indofood Sukses Makmur', 'sektor': 'Konsumen'},
    'MYOR.JK': {'nama': 'Mayora Indah', 'sektor': 'Konsumen'},
    'GGRM.JK': {'nama': 'Gudang Garam', 'sektor': 'Konsumen'},
    'HMSP.JK': {'nama': 'HM Sampoerna', 'sektor': 'Konsumen'},
    'KLBF.JK': {'nama': 'Kalbe Farma', 'sektor': 'Konsumen'},
    'AMRT.JK': {'nama': 'Sumber Alfaria Trijaya', 'sektor': 'Konsumen'},
    'CPIN.JK': {'nama': 'Charoen Pokphand', 'sektor': 'Konsumen'},
    'JPFA.JK': {'nama': 'Japfa Comfeed', 'sektor': 'Konsumen'},

    'ADRO.JK': {'nama': 'Adaro Energy', 'sektor': 'Energi'},
    'PTBA.JK': {'nama': 'Bukit Asam', 'sektor': 'Energi'},
    'ITMG.JK': {'nama': 'Indo Tambangraya Megah', 'sektor': 'Energi'},
    'HRUM.JK': {'nama': 'Harum Energy', 'sektor': 'Energi'},
    'MEDC.JK': {'nama': 'Medco Energi', 'sektor': 'Energi'},
    'AKRA.JK': {'nama': 'AKR Corporindo', 'sektor': 'Energi'},
    'UNTR.JK': {'nama': 'United Tractors', 'sektor': 'Energi'},
    'INDY.JK': {'nama': 'Indika Energy', 'sektor': 'Energi'},
    'MBMA.JK': {'nama': 'Merdeka Battery', 'sektor': 'Energi'},
    'ENRG.JK': {'nama': 'Energi Mega Persada', 'sektor': 'Energi'},
}