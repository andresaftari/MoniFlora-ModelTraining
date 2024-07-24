#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import requests
import json
import time
import os

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'

class config:
	key = 'a92d6963f9d077f844b11a9ea264579c'

def banner():
	os.system('clear')
	print(color.BLUE + '''
████████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗
   ██║   ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝
   ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗
   ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
	''' + color.END)

def main():
    banner()    
    if len(sys.argv) == 2:
        number = sys.argv[1]
        api = 'http://apilayer.net/api/validate?access_key=' + config.key + '&number=' + number + '&country_code=&format=3'
        output = requests.get(api)
        content = output.text
        obj = json.loads(content)
        print(content)
        country_code = obj['country_code']
        country_name = obj['country_name']
        location = obj['location']
        carrier = obj['carrier']
        line_type = obj['line_type']

        print ('--------------------------------------')
        time.sleep(0.2)
        
        success_status = '[ ' + color.GREEN + 'OK ' + color.END + ']'
        failure_status = '[ ' + color.RED + 'FAILED ' + color.END + ']'

        print(f' - Getting Country Code: {success_status if country_code else failure_status}')
        print(f' - Getting Country Name: {success_status if country_name else failure_status}')
        print(f' - Getting Location: {success_status if location else failure_status}')
        print(f' - Getting Carrier: {success_status if carrier else failure_status}')
        print(f' - Getting Device Type: {success_status if line_type else failure_status}')

        print('')
        print(color.BLUE + '[+] ' + color.END + 'Information Output')
        print('--------------------------------------')
        print(' - Phone number: ' +str(number))
        print(' - Country: ' +str(country_code))
        print(' - Country Name: ' +str(country_name) + f'{success_status if carrier else failure_status}')
        print(' - Location: ' +str(location) + f'{success_status if location else failure_status}')
        print(' - Carrier: ' +str(carrier) + f'{success_status if carrier else failure_status}')
        print(' - Device: ' +str(line_type) + f'{success_status if line_type else failure_status}')
    else:
        print ('[TRACKER] Usage:')
        print ('./%s <phone-number>' % (sys.argv[0]))
        print ('./%s +13213707446' % (sys.argv[0]))

main()