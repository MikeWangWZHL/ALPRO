########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

subscription_key = 'ada4e86a3dab4fc9a1513ab5b6003442' # Ruochen's api key

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': f'{subscription_key}',
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'en',
    'model-version': 'latest',
})

# from url
body_object = {
    "url":"https://cdn.pixabay.com/photo/2018/04/26/12/14/travel-3351825_960_720.jpg"
}
body = json.dumps(body_object)

# from file?
local_image_path = os.path.join (images_folder, "faces.jpg")
local_image = open(local_image_path, "rb")

conn = http.client.HTTPSConnection('westus2.api.cognitive.microsoft.com')
conn.request("POST", "/vision/v3.2/tag?%s" % params, "{}".format(body), headers)
response = conn.getresponse()
data = response.read()
print(data)
conn.close()

# try:
#     body_object = {
#         "url":"https://en.wikipedia.org/wiki/House#/media/File:103_Hanover.jpg"
#     }
#     body = json.dumps(body_object)

#     conn = http.client.HTTPSConnection('*.cognitiveservices.azure.com')
#     conn.request("POST", "/vision/v3.2/tag?%s" % params, f"{body}", headers)
#     response = conn.getresponse()
#     data = response.read()
#     print(data)
#     conn.close()
# except Exception as e:
#     print(e)

####################################