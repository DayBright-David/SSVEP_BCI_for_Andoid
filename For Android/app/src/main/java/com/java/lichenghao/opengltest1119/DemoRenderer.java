package com.java.lichenghao.opengltest1119;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.util.Log;
import android.widget.RelativeLayout;

public class DemoRenderer implements GLSurfaceView.Renderer {
    private GLSquare[] squares = new GLSquare[40];



    private final float[] mMVPMatrix = new float[16];      //变换矩阵（投影矩阵*相机矩阵的结果，最终要传递给顶点着色器）
    private final float[] mProjectionMatrix = new float[16]; //投影矩阵
    private final float[] mViewMatrix = new float[16]; //相机位置矩阵

    private float ratio = 1.0f;
    private long begin_time;
    private int red = 0;






    @Override
    public void onSurfaceCreated(GL10 unused, EGLConfig config) {

        //设置背景
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        begin_time = System.nanoTime();

        //初始化方块
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < 8; j ++){
                squares[i*8+j] = new GLSquare();
                squares[i*8+j].set(-0.93f + 0.28f * j ,
                        -0.53f + 0.28f * i ,
                        8.0f+ (j+0.2f*i),
                        0.175f,
                        0.5f * (i+j));
            }
        }


    }

    private FirstFragment firstFragment = null;
    public void setFirstFragment(FirstFragment fragment){
        firstFragment = fragment;
    }


    @Override
    public void onSurfaceChanged(GL10 unused, int width, int height) {
        // 设置绘图的窗口
        Looper.prepare();
        GLES20.glViewport(0,0,width,height);
        //获取绘图区域长宽比，用于修正
        ratio = (float) width/ height;

        Message message = Message.obtain();
        message.what = 20220708;
        message.arg1 = width;
        message.arg2 = height;

//        Messenger messenger = new Messenger( new Handler(){
//            @Override
//            public void handleMessage(final Message msg) {
//                    Log.d("Demo Renderer","ok");
//            }
//        });
//
//        try {
//            messenger.send(message);
//        } catch (RemoteException e) {
//            e.printStackTrace();
//        }
//        Handler handler = new Handler(){
//            @Override
//            public void handleMessage(Message msg){
//                super.handleMessage(msg);
//            }
//        };
//        handler.sendMessage(message);
        Log.d("renderer", width + " " + height);

        if(firstFragment != null){
            firstFragment.paintText(width, height);
        }

    }

    @Override
    public void onDrawFrame(GL10 unused) {


        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        //依次绘制
        for(int i = 0; i< 40 ; i++){
            squares[i].draw(ratio, begin_time, (i == red));
        }

    }

    public static int loadShader(int type, String shaderCode){

        // create a vertex shader type (GLES20.GL_VERTEX_SHADER)
        // or a fragment shader type (GLES20.GL_FRAGMENT_SHADER)
        int shader = GLES20.glCreateShader(type);

        // add the source code to the shader and compile it
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);

        return shader;
    }
}
