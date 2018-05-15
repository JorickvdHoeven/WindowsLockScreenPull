from datetime import datetime, timedelta
from shutil import copyfile
from hashlib import sha256
from os import path, makedirs, walk, getenv, rename
from imghdr import what
from time import sleep
from sys import argv
from PIL import Image
from requests import post, HTTPError
from pandas import DataFrame, concat, read_csv
try:
    from APIKey import AzureAPIKey
    Azure = True
except ImportError:
    AzureAPIKey = ''
    Azure = False


class Python_image_loader:

    def __init__(self, sourceDir, destDir):
        """ Initialise all of the variables and set the API Key
        """
        # 1. Initilise, grab variables
        self.subscription_key = AzureAPIKey
        self.vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v1.0/"
        self.vision_analyze_url = self.vision_base_url + "analyze"

        self.sourceDir = sourceDir
        self.destinationDir = destDir
        self._copyLogFile = self.destinationDir + 'copyLog.dat'

        # 2. Check for source and Destination folders (if no Dest, create dest. If not source, fail)
        if not path.exists(self.sourceDir):
            exit()

        if not path.exists(self.destinationDir):
            print('[+] Destination directory doesn''t exist, creating...')
            makedirs(self.destinationDir)

        # 3. Check for source log (If no source log, generate)
        if not path.exists(self._copyLogFile):
            self.__reset_copy_log()
        else:
            # 4. Load source log data into memory
            self.__load_copy_log(self._copyLogFile)

    def __reset_copy_log(self):
        """ Create a new log of all the files in the folder
        """
        print('[+] Rebuilding copy log... this may take a few mins')
        existingFiles = self.get_hashes(self.destinationDir, True)
        self._copyLog = existingFiles if not existingFiles.empty else DataFrame(columns=["SHA256", "Path", "Azure_Processed"])
        self._copyLog.to_csv(self._copyLogFile, quoting=1)
        print('[+] Copy Log Rebuilt')

    def __load_copy_log(self, logfile):
        """ Load an existing log of all the files in the folder
        """
        with open(logfile, 'r') as f:
            self._copyLog = read_csv(f, quoting=1, index_col=0)
        print('[!!] TODO: Perform Copy Log integrity Checks')
        # TODO: check that all of the files still exist, and if not find them

    # Hash all files in source
    def get_hashes(self, source, no_jpg_size_filter=True):
        """ Returns SHA256 hashes of files in the folder.
            Setting True to no_jpg_size_filter allows the function
            to hash all files, otherwise it will limit to jpg files of 1920X1080 resolution
        """
        BUF_SIZE = 65345
        hash_dict = DataFrame()
        for root, dirs, files in walk(source):
            for name in files:
                if what(path.join(root, name)) == 'jpeg' or no_jpg_size_filter:
                    im = Image.open(path.join(root, name))
                    if im.size == (1920, 1080) or no_jpg_size_filter:
                        with open(path.join(root, name), 'rb') as f:
                            sha1 = sha256()
                            while True:
                                data = f.read(BUF_SIZE)
                                if not data:
                                    break
                                sha1.update(data)
                            line = DataFrame([{"SHA256": sha1.hexdigest(), "Path": path.join(
                                root, name), "Azure_Processed": 0}])
                            hash_dict = concat(
                                [hash_dict, line], axis=0, ignore_index=True)
        # print(hash_dict.to_string())
        return hash_dict

    def azureVisionUpdate(self, image_path):
        """ This is a wrapper for the Azure Compute Vision API call
        """
        try:
            image_data = open(image_path, "rb").read()
        except:
            print("[!!] File not found: " + image_path)
            return path.basename(image_path)

        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key,
                   "Content-Type": "application/octet-stream"}
        params = {'visualFeatures': 'Categories,Description,Color'}
        try:
            response = post(self.vision_analyze_url,
                            headers=headers,
                            params=params,
                            data=image_data)
            response.raise_for_status()
            analysis = response.json()
        except HTTPError as e:
            print('[ERROR] HTTP Error: {}'.format(e.response.text))
            print('[+] Saving Copy Log and exiting... ')
            self._copyLog.to_csv(self._copyLogFile, quoting=1)
            exit()

        # print(analysis)
        try:
            image_caption = analysis["description"]["captions"][0]["text"].capitalize(
            )
            main_category = analysis["categories"]
        except:
            print("[!] error in the image caption") 
            print(analysis)
            image_caption = "ERROR_UNKNOWN"
            main_category = "UNKNOWN"

        if len(main_category) > 0:
            prefix = max(analysis["categories"],
                         key=lambda x: x['score'])['name']
        else:
            prefix = 'NoCat'
        
        return prefix.upper() + '_' + image_caption.replace(' ', '_') + '.jpg'

    def __unprocessedFileName(self, filename):
        if len(filename) == 68 and '_' not in filename:
            return True
        else:
            return False

    def findPics(self, Azure):
        # get the source Hashes
        print(
            '[+] Examining new files and checking for viable backgrounds (jpg, 1920x1080 only)')
        newHashes = self.get_hashes(self.sourceDir, False)
        # test for new ones
        newHashes = newHashes.merge(self._copyLog, on=['SHA256'],
                                    how='left', indicator=True)

        newHashes = newHashes.loc[newHashes['_merge'] == 'left_only']

        if newHashes.shape[0] > 0:
            print('[+] Copying {} files over'.format(newHashes.shape[0]))
            # move new ones to destination
            for index, row in newHashes.iterrows():
                copyfile(row['Path_x'], path.join(
                    self.destinationDir, path.basename(row['Path_x'] + '.jpg')))
                line = DataFrame([{"SHA256": row['SHA256'], "Path":path.join(
                    self.destinationDir, path.basename(row['Path_x'] + '.jpg')), "Azure_Processed":0}])
                self._copyLog = concat(
                    [self._copyLog, line], axis=0, ignore_index=True)

            # update the copyLog file given that we've updated the files
            self._copyLog.to_csv(self._copyLogFile, quoting=1)
        else:
            print('[+] No new files found')

        print('[+] Checking files for taggable files')
        # check jpeg files for wierd 65 char file names and update
        if Azure:
            for index, row in self._copyLog.loc[self._copyLog['Azure_Processed'] == 0].iterrows():
                if self.__unprocessedFileName(path.basename(row['Path'])):
                    print(
                        '[+] Starting Azure image tagging for {} ...'.format(path.basename(row['Path'])))
                    endTime = datetime.now() + timedelta(seconds=3)
                    newFileName = self.azureVisionUpdate(row['Path'])
                    newFileName = newFileName.replace('__', '_')
                    while datetime.now() < endTime:
                        sleep(0.5)
                    newPath = path.join(path.dirname(row['Path']), newFileName)
                    if path.isfile(newPath):
                        newPath = newPath[:-4] + ' (1)' + newPath[-4:]
                    try:
                        rename(row['Path'], newPath)
                        self._copyLog.at[index, 'Path'] = newPath
                        self._copyLog.at[index, 'Azure_Processed'] = 1
                        print(
                            '[+] Azure image tagging finished, new name: {}'.format(newFileName))
                    except OSError:
                        print('[ERROR] Image {} not updated'.format(
                            path.basename(row['Path'])))
        self._copyLog.to_csv(self._copyLogFile, quoting=1)
        print('[+] Copy Log Saved')


def main():
    if len(argv) < 2:
        # This may change between machines.. will need to be tested
        sourceDir = getenv(
            'LOCALAPPDATA') + "\\Packages\\Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy\\LocalState\\Assets\\"
        destinationDir = getenv('HOMEPATH') + "\\Pictures\\Windows Spotlight\\"
    else:
        if len(argv) == 2:
            sourceDir = argv[0]
            destinationDir = argv[0]
            if not path.isdir(sourceDir):
                print('[ERROR] Source directory doesn''t exist')
                exit()
        else:
            print(
                '[ERROR] wrong arguments given, format is: PythonImageLoader.py Source Destination')
            exit()

    pil = Python_image_loader(sourceDir, destinationDir)
    pil.findPics(Azure)
    print('[=] === Execution Finished ===')
    exit()


if __name__ == "__main__":
    main()
