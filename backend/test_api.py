import requests
import os
import base64


API_URL = "http://127.0.0.1:8000/analyze"


IMAGE_PATH = r"test_cxr.png"


def run_api_test():
    """
    Sends a test image to the running Rad-Insight API, prints the response,
    and saves the returned heatmap to a file.
    """
    print(f"--- Starting API Test ---")
    print(f"Targeting API endpoint: {API_URL}")
    print(f"Using image: {IMAGE_PATH}\n")

    
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Test image not found at '{IMAGE_PATH}'")
        print("Please make sure your test image is named 'test_cxr.png' and is in this directory.")
        return

    
    with open(IMAGE_PATH, "rb") as image_file:
        
        files = {"image": (os.path.basename(IMAGE_PATH), image_file, "image/png")}
        
        try:
            
            response = requests.post(API_URL, files=files, timeout=30) # 30-second timeout

            
            if response.status_code == 200:
                print("--- SUCCESS! API returned a valid response. ---\n")
                data = response.json()
                
                
                print("Probabilities:")
                if "probabilities" in data and data["probabilities"]:
                    for pathology, prob in data["probabilities"].items():
                        print(f"  - {pathology:<20}: {prob:.4f}")
                else:
                    print("  - No probabilities returned.")

                
                print("\nHeatmap Status:")
                if "heatmap_image" in data and data["heatmap_image"]:
                    print(f"  - Heatmap data received successfully.")
                    
                    
                    try:
                        heatmap_b64_string = data["heatmap_image"]
                        
                        if "," in heatmap_b64_string:
                            header, encoded = heatmap_b64_string.split(",", 1)
                        else:
                            encoded = heatmap_b64_string
                        
                        image_data = base64.b64decode(encoded)
                        
                        output_filename = "heatmap_from_api.png"
                        with open(output_filename, "wb") as f:
                            f.write(image_data)
                        print(f"  - Heatmap visually saved to '{output_filename}' for inspection.")
                            
                    except Exception as e:
                        print(f"  - ERROR: Could not save the heatmap file. Reason: {e}")
                else:
                    print("  - Heatmap not generated or returned.")

            else:
                print(f"--- FAILURE! API returned an error. ---")
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")

        except requests.exceptions.ConnectionError as e:
            print(f"--- FAILURE! Could not connect to the API. ---")
            print(f"Error: {e}")
            print("Is the FastAPI server running? Start it with: uvicorn main:app --reload")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_api_test()