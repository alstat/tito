import os
import base64

def handle_zip(data):
    """
    Helper Function for Handling the ZIP Files
    """
    data = data.partition(",")[2]          # strip the header
    data = data + '=' * (-len(data) % 4)   # add padding
    zip_file = base64.b64decode(data)      # decode the base64 string

    filename = "decoded_zip.zip"           # default filename
    with open(filename, "wb") as f:        # write the file as a zip file again
        f.write(zip_file)
    
    return {"response":"Hey, here is the response"}