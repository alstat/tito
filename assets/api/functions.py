import os
import base64
import zipfile

def decode(data):
    """
    Helper Function for Decoding the Files
    """
    data = data.partition(",")[2]          # strip the header
    data = data + '=' * (-len(data) % 4)   # add padding
    return base64.b64decode(data)          # decode the base64 string

def save(data, path = os.getcwd(), name = "decoded_file", fext = ".zip"):
    """
    Helper Function for Saving the ZIP File

    # Arguments:
        data: string, base 64 encoded file
        path: string, default: current working directory, directory where the zip file will be saved
        name: string, default: "decoded_file", zip filename with no file extension
        fext: string, default: ".zip", file extension
    """
    inp_file = decode(data)

    f = path + "/data/" + name + fext            # default filename
    with open(f, "wb") as d:                     # write the file as a zip file again
        d.write(inp_file)

def extract(inp_path, out_path = os.getcwd(), name = "decoded_file"):
    """
    Helper Function for Extracting the ZIP File

    # Arguments:
        inp_path: string, directory where the zip file is saved
        out_path: string, default: current directory, directory where the unzip file will be saved
        name    : string, default: "decoded_file", zip filename with no file extension
    """
    zip_data = zipfile.ZipFile(inp_path + "/data/" + name + ".zip", "r")
    zip_data.extractall(out_path + "/data/" + name)
    zip_data.close()

def handle_zip(data, filename = "decoded_file"):
    """
    Helper Function for Handling ZIP File

    # Arguments:
        data: string, base 64 encoded file
    """

    save(data, name = filename)
    extract(os.getcwd(), name = filename)
    return {"response" : "Hey, here is the response"}

def train(args):
    return args

def analyze(args):
    return args