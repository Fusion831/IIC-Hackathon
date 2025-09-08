import requests
import os
import base64

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/analyze"
IMAGE_PATH = r"test_cxr.png"

def run_api_test():
    """
    Sends a test image to the Rad-Insight API and prints the full response,
    including probabilities, heatmap status, and the NLP-generated report.
    """
    print(f"--- Starting API Test ---")
    print(f"Targeting API endpoint: {API_URL}")
    print(f"Using image: {IMAGE_PATH}\n")

    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Test image not found at '{IMAGE_PATH}'")
        print("Please place a test image named 'test_cxr.png' in this directory.")
        return

    # Prepare the file for multipart upload
    with open(IMAGE_PATH, "rb") as image_file:
        files = {"image": (os.path.basename(IMAGE_PATH), image_file, "image/png")}

        try:
            # Send the request to the API
            response = requests.post(API_URL, files=files, timeout=45) # Increased timeout for NLP

            # --- Process the Response ---
            if response.status_code == 200:
                print("--- SUCCESS! API returned a valid response. ---\n")
                data = response.json()
                
                # 1. Print Probabilities
                print("Probabilities:")
                if "probabilities" in data and data["probabilities"]:
                    for pathology, prob in data["probabilities"].items():
                        print(f"  - {pathology:<20}: {prob:.4f}")
                else:
                    print("  - No probabilities returned.")

                # 2. Print Heatmap Status
                print("\nHeatmap Status:")
                if "heatmap_image" in data and data["heatmap_image"]:
                    print(f"  - Heatmap data received successfully.")
                else:
                    print("  - Heatmap not generated or returned.")

                # 3. Print AI-Generated Report
                print("\nAI-Generated Radiology Report:")
                if "report_text" in data and data["report_text"]:
                    print("-----------------------------------------")
                    print(data["report_text"])
                    print("-----------------------------------------")
                else:
                    print("  - Report not generated or returned.")

            else:
                print(f"--- FAILURE! API returned an error. ---")
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")

        except requests.exceptions.ConnectionError as e:
            print(f"--- FAILURE! Could not connect to the API. ---")
            print(f"Error: {e}")
            print("Is the FastAPI server running? Start it with: uvicorn main:app --reload")
        except requests.exceptions.ReadTimeout:
             print(f"--- FAILURE! The request timed out. ---")
             print("The server is taking too long to respond. This can happen if the NLP model is slow.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_api_test()

