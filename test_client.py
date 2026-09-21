import requests

def ask_question(question):
    url = "http://127.0.0.1:5000/query"
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        print("\n=== Query ===")
        print(question)

        print("\n=== Answer ===")
        print(data.get("answer", "No answer"))

        print("\n=== Sources ===")
        print(data.get("sources", []))

        print("\n=== Excerpts (retrieved chunks) ===")
        excerpts = data.get("excerpts", {})
        if excerpts:
            for src, text in excerpts.items():
                print(f"\nFrom {src}:\n{text[:300]}...")  # show first 300 chars
        else:
            print("No excerpts retrieved")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # 🔎 Try with known terms from your PDFs
    ask_question("Pregnancy Testing")
    ask_question("Medical Response Plan")
    ask_question("Routing Slip submission")

