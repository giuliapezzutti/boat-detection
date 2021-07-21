import os
import glob
import xml.etree.ElementTree as ET

def xml_to_string(xml_filename, txt_file):
# xml_filename is a string, txt_file is an object
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    for fn in root.iter('filename'): fname = fn.text
    fname = fname + '\n'
    txt_file.write(fname)

    labels = [label.text for label in root.iter('name')]
    xmins = [val.text for val in root.iter('xmin')]
    xmaxs = [val.text for val in root.iter('xmax')]
    ymins = [val.text for val in root.iter('ymin')]
    ymaxs = [val.text for val in root.iter('ymax')]

    for i in range(0, len(xmins)):
        line = labels[i] + ':'
        bndbox = [xmins[i], xmaxs[i], ymins[i], ymaxs[i]]
        bndbox_str = ';'.join(str(k) for k in bndbox)
        line = line + bndbox_str + ';\n'
        txt_file.write(line)

working_dir = os.getcwd()
labels_dir_name = 'DATASET_LABELS'
print('Specify the name of the directory containing .xml files (press ENTER for default name "' + labels_dir_name + '"): ')
input_ = input()
if input_ != '': labels_dir_name = input_
labels_dir = working_dir + '/' + labels_dir_name
os.chdir(labels_dir)
filenames = []
for file in glob.glob("*.xml"):
    filenames.append(file)
filenames.sort()

labels_txt = labels_dir_name + '.txt'
text_file = open(labels_txt, "w")  # option "a" to append to existing file
text_file.close()
text_file = open(labels_txt, "a")
for filename in filenames:
    xml_to_string(filename, text_file)
text_file.close()

src = os.getcwd() + '/' + labels_txt
dst = working_dir + '/' + labels_txt
os.rename(src, dst)

print(labels_dir_name + '.txt is ready!')