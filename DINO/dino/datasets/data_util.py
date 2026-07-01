import os
import os.path as osp
import shutil
import time
import datetime

import torch

# [MODIFICADO]: Eliminada la dependencia de SLConfig
# from util.slconfig import SLConfig

class Error(OSError):
    pass

def slcopytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    """
    modified from shutil.copytree without copystat.
    """
    errors =[]
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        os.symlink(linkto, dstname)
                    else:
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        if os.path.isdir(srcname):
                            slcopytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    slcopytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    copy_function(srcname, dstname)
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return slcopytree(src_path, tgt_path)


def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)


def preparing_dataset(pathdict, image_set, args):
    #[MODIFICADO]: Vaciamos esta función ya que era exclusiva para clústeres de Meta/Facebook
    # y dependía de SLConfig y static_data_path.py. Nosotros usamos nuestro propio data_loader.py
    return None