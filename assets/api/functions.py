import os, time
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
    # [TITO] Pre-processing of profile images
    facenetLib = 'assets/facenet-for-tito/'
    import subprocess
    filename = "data\decoded_file"
    
    # Pre-processing
    print('Pre-processing . . . ')
    tito_preprocess_profiles()
    print('Pre-processing completed.')

    # Training
    print('Training . . . ')
    batch_accuracy, batch_runtime = tito_train()
    print('Training completed.')

    results = {}
    results["accuracy"] = batch_accuracy
    results["runtime"] = batch_runtime

    print(results)
    return results 
    #"{accuracy: "+ str(batch_accuracy) + ", runtime: " + str(batch_runtime)+"}"

def tito_preprocess_profiles():
    '''
    '''
    filepath = os.path.realpath(__file__)
    filepath = os.path.dirname(filepath)
    input_dir = filepath+'\..\..\data\decoded_file\profiles'
    output_dir = filepath+'\..\..\output\profiles_inter'
    crop_dim = 180

    from tito_facenet import preprocess

    #preprocess.predefine()
    preprocess.main(input_dir, output_dir, crop_dim)

def tito_train():
    '''
    '''
    from tito_facenet import use_classifier as tito_classify

    filepath = os.path.realpath(__file__)
    filepath = os.path.dirname(filepath)

    facenetLib = filepath+'/../../assets/api/tito_facenet'
    log_dir = 'lfw'
    input_dir = filepath+'\..\..\output\profiles_inter'
    classifier_path = filepath+'\..\..\output\model\classifier_test.pkl'
    model_path = facenetLib + '\etc/20170511-185253/20170511-185253.pb'
    num_threads = 16
    num_epochs = 5
    min_images_per_class = 5
    batch_size = 128
    split_ratio = 0.8
    
    if not os.path.exists(filepath+'/../../output/'+log_dir):
        os.makedirs(filepath+'/../../output/'+log_dir)

    # Data preparation: splitting of training and test set
    train, test = tito_classify._data_prep(   input_directory=input_dir, 
                                min_images_per_labels=min_images_per_class, 
                                split_ratio=split_ratio)
    
    # Training of the model: set is_train=True
    xx, batch_runtime = tito_classify.main(   dataset=train, input_directory=input_dir, 
                                model_path=model_path, 
                                classifier_output_path=classifier_path,
                                batch_size=batch_size,
                                num_threads=num_threads,
                                num_epochs=num_epochs,
                                min_images_per_labels=min_images_per_class,
                                split_ratio=split_ratio, is_train=True)

    # Testing of the model: set is_train=False
    batch_accuracy, xx = tito_classify.main(  dataset=test, input_directory=input_dir, 
                                model_path=model_path, 
                                classifier_output_path=classifier_path,
                                batch_size=batch_size,
                                num_threads=num_threads,
                                num_epochs=num_epochs,
                                min_images_per_labels=min_images_per_class,
                                split_ratio=split_ratio, is_train=False)

    return batch_accuracy, batch_runtime



def analyze(args):
    return args