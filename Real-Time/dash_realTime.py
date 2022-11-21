import streamlit as st
import requests
from json import loads, dumps
from time import sleep


def get_price() :
    data = loads(
            requests.get(
                'http://api.coincap.io/v2/assets/bitcoin'
            ).content.decode()
    )
    return round(float(data['data']['priceUsd']),2)


placeholder = st.empty()

cur = get_price()

for i in range(200) :

    prev = cur
    cur = get_price()

    with placeholder.container() :
        st.metric(
            label='Bitcoin USD ($)',
            value=cur,
            delta=round(cur-prev,2)
        )

    sleep(10)


