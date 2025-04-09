import os
import sys
import argparse
from subprocess import Popen, PIPE
import fcntl
import time
import traceback

"""
Module adapted from https://github.com/mcarans/resetusb/
"""

path, name, name_list = None, None, None

parser = argparse.ArgumentParser(description='Reboots USB, PCI devices')
parser.add_argument('--path',
                    help="Specify the path to the USB or PCI device")
parser.add_argument('--name', type=str,
                    help="Name of the device to be reset, either Partial or Complete Correct name")
parser.add_argument('--name_list', type=str,
                    help="List of device names to be reset, either Partial or Complete Correct name")
args = parser.parse_args()


def create_pci_list():
    pci_usb_list = list()
    try:
        lspci_out = Popen('lspci -Dvmm', shell=True, bufsize=64, stdin=PIPE, stdout=PIPE, close_fds=True).stdout.read().strip().decode('utf-8')
        pci_devices = lspci_out.split('%s%s' % (os.linesep, os.linesep))
        for pci_device in pci_devices:
            device_dict = dict()
            categories = pci_device.split(os.linesep)
            for category in categories:
                key, value = category.split('\t')
                device_dict[key[:-1]] = value.strip()
            if 'USB' not in device_dict['Class']:
                continue
            for root, dirs, files in os.walk('/sys/bus/pci/drivers/'):
                slot = device_dict['Slot']
                if slot in dirs:
                    device_dict['path'] = os.path.join(root, slot)
                    break
            pci_usb_list.append(device_dict)
    except Exception as ex:
        print('Failed to list pci devices! Error: %s' % ex)
        # sys.exit(-1)
    return pci_usb_list


def create_usb_list():
    device_list = list()
    try:
        lsusb_out = Popen('lsusb -v', shell=True, bufsize=64, stdin=PIPE, stdout=PIPE, close_fds=True).stdout.read().strip().decode('utf-8')
        usb_devices = lsusb_out.split('%s%s' % (os.linesep, os.linesep))
        for device_categories in usb_devices:
            if not device_categories:
                continue
            categories = device_categories.split(os.linesep)
            device_stuff = categories[0].strip().split()
            bus = device_stuff[1]
            device = device_stuff[3][:-1]
            device_dict = {'bus': bus, 'device': device}
            device_info = ' '.join(device_stuff[6:])
            device_dict['description'] = device_info
            for category in categories:
                if not category:
                    continue
                categoryinfo = category.strip().split()
                if categoryinfo[0] == 'iManufacturer':
                    manufacturer_info = ' '.join(categoryinfo[2:])
                    device_dict['manufacturer'] = manufacturer_info
                if categoryinfo[0] == 'iProduct':
                    device_info = ' '.join(categoryinfo[2:])
                    device_dict['device'] = device_info
            path = '/dev/bus/usb/%s/%s' % (bus, device)
            device_dict['path'] = path

            device_list.append(device_dict)
    except Exception as ex:
        print('Failed to list usb devices! Error: %s' % ex)
        # sys.exit(-1)
    return device_list


def print_usb_list():
    usb_list = create_usb_list()
    for device in usb_list:
        print('path=%s' % device['path'])
        print('    description=%s' % device['description'])
        print('    manufacturer=%s' % device['manufacturer'])
        print('    device=%s' % device['device'])
        print('    search string=%s %s %s' % (device['description'], device['manufacturer'], device['device']))


def print_pci_list():
    pci_usb_list = create_pci_list()
    for device in pci_usb_list:
        print('path=%s' % device['path'])
        print('    manufacturer=%s' % device['SVendor'])
        print('    device=%s' % device['SDevice'])
        print('    search string=%s %s' % (device['SVendor'], device['SDevice']))


def reset_pci_usb_device(dev_path):
    folder, slot = os.path.split(dev_path)
    try:
        fp = open(os.path.join(folder, 'unbind'), 'wt')
        fp.write(slot)
        fp.close()
        fp = open(os.path.join(folder, 'bind'), 'wt')
        fp.write(slot)
        fp.close()
        print('Successfully reset %s' % dev_path)
        # sys.exit(0)
    except Exception as ex:
        print('Failed to reset device! Error: %s' % ex)
        # sys.exit(-1)


def reset_usb_device(dev_path):
    USBDEVFS_RESET = 21780
    try:
        f = open(dev_path, 'w', os.O_WRONLY)
        fcntl.ioctl(f, USBDEVFS_RESET, 0)
        print('Successfully reset %s' % dev_path)
        # sys.exit(0)
    except Exception as ex:
        print('Failed to reset device! Error: %s' % ex)
        # sys.exit(-1)

def reset_name_list(name_list_at):
    while True:
        try:
            name_list = name_list_at.split(",")
            print("namelist",name_list)
            pci_usb_list = create_usb_list()
            reset_list = []
            for item in name_list:
                for device in pci_usb_list:
                    if item in f"{device['description']} {device['manufacturer']} {device['device']}":
                        print(f"{device['description']} {device['manufacturer']} {device['device']}")
                        reset_usb_device(device['path'])
                        reset_list.append(item)
                        time.sleep(1)
            print(f"reset list {reset_list}")
            if len(reset_list) == 1 or len(reset_list) == 0 or (len(reset_list)==2 and ('455' not in reset_list)):
                print("all devices are not reset ! trying again ! check connection")
                time.sleep(1)
                continue
            else:
                print("all devices reset")
                break
        except Exception:
            print("error in resetting name list")
            traceback.print_exc()
        time.sleep(1)

path = args.path
name= args.name
name_list = args.name_list

print_usb_list()

if path:
    reset_usb_device(path)
else:
    pass

if name:
    temp_count = 0
    pci_usb_list = create_usb_list()
    for device in pci_usb_list:
        text = f"{device['description']} {device['manufacturer']} {device['device']}"
        if name in text:
            print("Device Found")
            reset_usb_device(device['path'])
            temp_count = 1
            break
    if temp_count == 0:
        print("Device not Found")
    print('Scan completed!')

if name_list:
    reset_name_list(name_list)
