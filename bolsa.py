#pip install streamlit
#pip install plotly
#pip install yfinance
#pip install Prophet

import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

tickers= {
    "VALE3.SA": "Vale",
    "PRIO3.SA": "Gás Petrobras",
    "ALOS3.SA": "Allos",
    "AZUL4.SA": "Azul",
    "GOOG": "Google",
    "EURUSD=X": "Euro Dolar",
    "GBPUSD=X": "GBP/USD",
    "CAD=X": "Canadian Dollar",
    "JPY=X": "US Dollar vs Japanese"
}

def carregar_dados(ticker, dt_inicial, dt_final):
    df = yf.Ticker(ticker).history(start= dt_inicial.strftime("%Y-%m-%d"),
                                   end= dt_final.strftime("%Y-%m-%d"))
    return df

def prever_dados(df, periodo):
    df.reset_index(inplace=True)
    df = df.loc[:, ['Date', 'Close']]
    df['Date']= df['Date'].dt.tz_localize(None)
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    
    modelo = Prophet()
    modelo.fit(df)
    
    datas_futuras= modelo.make_future_dataframe(periods=int(periodo)* 30)
    previsoes = modelo.predict(datas_futuras)
    return modelo, previsoes

st.image("logo_falcon.jpg")

lista_tickers = ["VALE3.SA", "PRIO3.SA", "ALOS3.SA", "AZUL4.SA",
                 "GOOG", "EURUSD=X", "GBPUSD=X", "CAD=X", "JPY=X"]

st.title("Minhas Previsões")

with st.sidebar:
    st.header("Menu Ações")
    ticker = st.selectbox("Selecione a Ação:", lista_tickers)
    dt_inicial = st.date_input("Data Inicial", value=date(2020, 1, 1))
    dt_final = st.date_input("Data Final")
    meses = st.number_input("Meses de Previsão", 1, 24, value=8)

dados = carregar_dados(ticker, dt_inicial, dt_final)

if dados.shape[0] != 0:
    st.header(f"Dados da Ação - {tickers[ticker]}")
    st.dataframe(dados)
    dados.head()

    st.subheader("Variação do Período")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name="Close"))
    st.plotly_chart(fig)


    st.header(f"Previsão Próximo(s) {meses} meses")
    modelo, previsoes = prever_dados(dados, meses)
    fig = plot_plotly(modelo, previsoes, xlabel="Período", ylabel="Valor")
    st.plotly_chart(fig)
else:
    st.warning("Nenhum Dado Encontrado no Periodo Selecionado")


# PARA EXECUTAR NO TERMINAL
# streamlit run bolsa.py
# source venv/scripts/activate - Ambiente Virtual
