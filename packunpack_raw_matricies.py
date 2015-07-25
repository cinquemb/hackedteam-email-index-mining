import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
import os
from time import sleep
from subprocess import Popen
import subprocess
import codecs
import re

is_unpack = True

def compress_file(file_):
	print "compressing:", file_
	cmd = 'tar -czf %s.tar.gz %s' % (file_, file_)
	try:
		proc = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=None, close_fds=True)
		proc_data = proc.communicate()[0]
		print proc_data
		data = True
	except Exception, e:
		data = False

	return data

def uncompress_file(file_):
    print "unpacking:", file_
    cmd = 'tar -xzf %s' % (file_)
    try:
        proc = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=None, close_fds=True)
        proc_data = proc.communicate()[0]
        print proc_data
        data = True
    except Exception, e:
        data = False

    return data

out_files = []
failed_dirs = []
def list_files(dir_,is_unpack=False):
    basedir = dir_
    tmp_path = os.path.abspath(dir_)
    if os.path.isdir(tmp_path):
        #print "Files in %s: " % (tmp_path)
        subdirlist = []
        try:
            files = os.listdir(tmp_path)
        except:
            failed_dirs.append(tmp_path)
            files = []

        for item in files:
            if os.path.isfile(item):
                print item
            else:
                subdirlist.append(os.path.join(basedir, item))

        for subdir in subdirlist:
            list_files(subdir, is_unpack)
    else:
        if not is_unpack:
            if 'tar.gz' not in dir_.split('/')[-1] and '.DS_Store' not in dir_.split('/')[-1]:
                out_files.append(u'%s' % (dir_))
        else:
            if 'tar.gz' in dir_.split('/')[-1] and '.DS_Store' not in dir_.split('/')[-1]:
                out_files.append(u'%s' % (dir_))


def check_if_compressed_exists(file_):
	tgz_file_ = file_ + '.tar.gz'
	if os.path.isfile(tgz_file_):
		return True
	else:
		return False


if __name__ == "__main__":
    list_files("raw_matrices/", is_unpack)
    if is_unpack:
        for x in out_files:
            uncompress_file(x)
    else:
    	for x in out_files:
    		if not check_if_compressed_exists(x):
    			compress_file(x)
