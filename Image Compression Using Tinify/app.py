import sys
import tinify
import os.path as path
tinify.key=''# Receive an API key from Tinify

def main():
    source=sys.argv[1]
    original_File=tinify.from_file(sys.argv[1])
    original_File.to_file('optimised.jpg')

    orignal_size=path.getsize(source)
    compressed_size=path.getsize('optimised.jpg')
    ratio=int((orignal_size/compressed_size)*10)
    print("Compression Rate: {}%".format(ratio))

if __name__ == "__main__":
    main()