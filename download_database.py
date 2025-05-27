import gdown


file_id = '1mekls6OGOKLmt7gYtHs0WGf5oTamTNat'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file and save it as 'downloaded_file.zip'
gdown.download(url, 'database.db', quiet=False)
