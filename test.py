import requests

#Replace with the server's actual IP and port
url = "http://127.0.0.1:8000/process"

#Path to the image you want to upload
image_file = "assets/framework.png"  # Replace with the path to your image file

#Open the image file and send it as part of a POST request
with open(image_file, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

#Print the response from the server
if response.status_code == 200:
    print("Label:", response.json())
else:
    print("Failed to process image:", response.text)
