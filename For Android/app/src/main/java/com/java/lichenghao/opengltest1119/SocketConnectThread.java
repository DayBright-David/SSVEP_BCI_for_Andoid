package com.java.lichenghao.opengltest1119;

import android.util.Log;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

class SocketConnectThread extends Thread{

    private static final String TAG = "SocketConnectThread";
    private Socket mSocket;
    private OutputStream mOutStream;
    private InputStream mInStream;
    private String mIpAddress;
    private int mClientPort;
    final byte[] buffer = new byte[2048];

    private void handleMessage(int len){
        Log.i(TAG + "::handleMessage", new String(buffer, 0, len));
    }

    public void writeMessage(String message){
        if(message.length() <= 0){
            return;
        }
        if(this.mOutStream != null){
            try{
                mOutStream.write(message.getBytes());
                mOutStream.flush();
            }catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }else{
            Log.e(TAG + "::writeMessage", "Try to write into a null out stream!");
        }
    }

    public void run(){
        try {
            //指定ip地址和端口号
            mSocket = new Socket(mIpAddress,mClientPort);
            if(mSocket != null){
                //获取输出流
                mOutStream = mSocket.getOutputStream();
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        Log.i(TAG,"connect success");

        try {
            while(true){
                mInStream = mSocket.getInputStream();
                this.handleMessage(mInStream.read(buffer));
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
    }
    public void closeConnection(){
        try{
            if(mOutStream != null){
                mOutStream.close();
                mOutStream = null;
            }
            if(mInStream != null){
                mInStream.close();
                mInStream = null;
            }
            if(mSocket != null){
                mSocket.close();
                mSocket = null;
            }
        }catch (Exception e){
            e.printStackTrace();
            return;
        }
    }


}