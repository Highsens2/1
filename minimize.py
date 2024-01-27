_C=None
_B=False
_A=True
import os,pandas
os.system('cls')
import pyfiglet,onnxruntime as ort,numpy as np,gc,numpy as np,cv2,time,win32api,win32con
from utils.general import cv2,non_max_suppression,xyxy2xywh
import torch
text='HIGHSENS AI'
ascii_art=pyfiglet.figlet_format(text,font='slant')
print(ascii_art)
print('Please Select Your Game Window')
print(' ')



screenShotHeight=320
screenShotWidth=320
useMask=_A
maskSide='left'
maskWidth=99
maskHeight=222
aaMovementAmp=.43
confidence=.734
aaQuitKey='L'
headshot_mode=_A
cpsDisplay=_A
visuals=_B
centerOfScreen=_A
onnxChoice=2





import pygetwindow,time,bettercam
from typing import Union
screenShotHeight,screenShotWidth
def gameSelection():
	K='Failed to activate game window: {}'
	try:
		F=pygetwindow.getAllWindows();print('=== All Windows ===')
		for(L,G)in enumerate(F):
			if G.title!='':print('[{}] {}'.format(L,G.title))
		try:M=int(input('Select the game you want to aimbot in: '))
		except ValueError:print("You didn't enter a valid number. Please try again.");return
		A=F[M]
	except Exception as C:print('Failed to select game window: {}'.format(C));return
	B=30;D=_B
	while B>0:
		try:A.activate();D=_A;break
		except pygetwindow.PyGetWindowException as N:print(K.format(str(N)));print('Trying again... (you should switch to the game now)')
		except Exception as C:print(K.format(str(C)));print('Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow');D=_B;B=0;break
		time.sleep(3.);B=B-1
	if D==_B:return
	print('Successfully activated the game window...');H=(A.left+A.right)//2-screenShotWidth//2;I=A.top+(A.height-screenShotHeight)//2;O,P=H+screenShotWidth,I+screenShotHeight;J=H,I,O,P;Q=screenShotWidth//2;R=screenShotHeight//2;print(J);E=bettercam.create(region=J,output_color='BGRA',max_buffer_len=512)
	if E is _C:print('Your Camera Failed! ');return
	E.start(target_fps=240,video_mode=_A);return E,Q,R
def main():
	l='dist';k='dist_from_center';j='confidence';i='height';h='width';g='current_mid_y';f='current_mid_x';e='images';J,K,L=gameSelection();I=0;M=time.time();F=''
	if onnxChoice==1:F='CPUExecutionProvider'
	elif onnxChoice==2:F='DmlExecutionProvider'
	elif onnxChoice==3:import cupy as m;F='CUDAExecutionProvider'
	N=ort.SessionOptions();N.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED;O=ort.InferenceSession('best.onnx',sess_options=N,providers=[F]);P=np.random.uniform(0,255,size=(1500,1));E=_C
	while win32api.GetAsyncKeyState(ord(aaQuitKey))==0:
		C=np.array(J.get_latest_frame())
		if useMask:
			maskSide.lower()
			if maskSide=='right':C[-maskHeight:,-maskWidth:,:]=0
			elif maskSide=='left':C[-maskHeight:,:maskWidth,:]=0
			else:raise Exception('ERROR: Invalid maskSide! Please use "left" or "right"')
		if onnxChoice==3:
			A=torch.from_numpy(C).to('cuda')
			if A.shape[2]==4:A=A[:,:,:3]
			A=torch.movedim(A,2,0);A=A.half();A/=255
			if len(A.shape)==3:A=A[_C]
		else:
			A=np.array([C])
			if A.shape[3]==4:A=A[:,:,:,:3]
			A=A/255;A=A.astype(np.half);A=np.moveaxis(A,3,1)
		if onnxChoice==3:Q=O.run(_C,{e:m.asnumpy(A)})
		else:Q=O.run(_C,{e:np.array(A)})
		A=torch.from_numpy(Q[0]).to('cpu');n=non_max_suppression(A,confidence,confidence,0,_B,max_det=10);B=[]
		for(D,G)in enumerate(n):
			o='';p=torch.tensor(A.shape)[[0,0,0,0]]
			if len(G):
				for R in G[:,-1].unique():q=(G[:,-1]==R).sum();o+=f"{q} {int(R)}, "
				for(*r,s,x)in reversed(G):B.append((xyxy2xywh(torch.tensor(r).view(1,4))/p).view(-1).tolist()+[float(s)])
		B=pandas.DataFrame(B,columns=[f,g,h,i,j]);S=[K,L]
		if len(B)>0:
			if centerOfScreen:B[k]=np.sqrt((B.current_mid_x-S[0])**2+(B.current_mid_y-S[1])**2);B=B.sort_values(k)
			if E:B['last_mid_x']=E[0];B['last_mid_y']=E[1];B[l]=np.linalg.norm(B.iloc[:,[0,1]].values-B.iloc[:,[4,5]],axis=1);B.sort_values(by=l,ascending=_B)
			T=B.iloc[0].current_mid_x;U=B.iloc[0].current_mid_y;V=B.iloc[0].height
			if headshot_mode:W=V*.37
			else:W=V*.25
			X=[T-K,U-W-L]
			if win32api.GetKeyState(20):win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,int(X[0]*aaMovementAmp),int(X[1]*aaMovementAmp),0,0)
			E=[T,U]
		else:E=_C
		if visuals:
			for D in range(0,len(B)):Y=round(B[h][D]/2);Z=round(B[i][D]/2);a=B[f][D];b=B[g][D];c,H,t,u=int(a+Y),int(b+Z),int(a-Y),int(b-Z);d=0;v='{}: {:.2f}%'.format('Player',B[j][D]*100);cv2.rectangle(C,(c,H),(t,u),P[d],2);w=H-15 if H-15>15 else H+15;cv2.putText(C,v,(c,w),cv2.FONT_HERSHEY_SIMPLEX,.5,P[d],2)
		I+=1
		if time.time()-M>1:
			if cpsDisplay:print('CPS: {}'.format(I))
			I=0;M=time.time();gc.enable()
		if visuals:
			cv2.imshow('View',C)
			if cv2.waitKey(1)&255==ord('q'):exit()
	J.stop()
if __name__=='__main__':
	try:main()
	except Exception as e:import traceback;traceback.print_exception(e);print('ERROR: '+str(e));print('')