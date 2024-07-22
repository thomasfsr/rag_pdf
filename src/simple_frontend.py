import streamlit as st
import httpx
import asyncio

FASTAPI_URL = "http://fastapi:8000/query"

def run():
    # col1, col2 = st.columns([.2,.8], vertical_alignment='center')
    # with col1:
    #     st.image('data/cover.png', width=100)
    # with col2:
    st.title("Consult saved PDFs")

    query_text = st.text_input("Enter your query:")

    async def get_response(query_text):
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", FASTAPI_URL, json={"query_text": query_text}) as response:
                response_text = ""
                async for chunk in response.aiter_text():
                    response_text += chunk
                    st.write(chunk)  # Stream response chunks to the app
                return response_text

    if st.button("Submit"):
        if query_text:
            response = asyncio.run(get_response(query_text))
            if response:
                st.success("Success!")
            else:
                st.error("Error: " + response)
        else:
            st.warning("Please enter a query.")