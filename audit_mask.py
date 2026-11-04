from pathlib import Path
import csv
import cv2
import numpy as np
from PIL import Image

def periksa_mask(path):
    with Image.open(path) as image:
        a=np.asarray(image.convert('RGB')).copy()
    h,w=a.shape[:2]
    if min(h,w)<12 or a.std(axis=(0,1)).max()<1:
        return Image.fromarray(a),'periksa: terlalu kecil atau seragam',0.
    k=max(1,round(.04*min(h,w)))
    m=np.zeros((h,w),np.uint8)
    cv2.setRNGSeed(42)
    cv2.grabCut(a,m,(k,k,w-2*k,h-2*k),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
    binary=np.uint8((m==1)|(m==3))
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    fraction=float(binary.mean())
    significant=sum(cv2.contourArea(c)>.02*w*h for c in contours)
    reasons=[]
    if not .03<=fraction<=.95: reasons.append('luas mask')
    if significant!=1: reasons.append('jumlah komponen')
    if not binary[h//2,w//2]: reasons.append('pusat di luar mask')
    a[binary==0]=(a[binary==0]*.35).astype(np.uint8)
    cv2.drawContours(a,contours,-1,(0,255,255),1)
    return Image.fromarray(a),('periksa: '+', '.join(reasons) if reasons else 'lolos pemeriksaan dasar'),fraction

def audit_folder(paths, destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'audit.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['crop','overlay','status','proporsi_mask','penilaian_manual'])
        for i,path in enumerate(paths):
            try:
                overlay,status,ratio=periksa_mask(path)
                filename=f'{i:05d}.jpg';overlay.save(dest/filename)
                writer.writerow([path,filename,status,ratio,''])
            except (OSError,ValueError,cv2.error) as e:
                writer.writerow([path,'','gagal: '+str(e),'',''])
