package com.java.lichenghao.opengltest1119;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.java.lichenghao.opengltest1119.databinding.FragmentFirstBinding;

import org.w3c.dom.Text;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;



public class FirstFragment extends Fragment {

    private FragmentFirstBinding binding;
    private Handler handler;




    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {

        binding = FragmentFirstBinding.inflate(inflater, container, false);
        TextView textView = binding.topText;
        textView.setText("asdf");

        return binding.getRoot();

    }

    public void paintText(int width, int height){

        Log.d("paint", width + " " + height);

        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                for(int j = 0; j < 8; j++){
                    for(int i = 0; i < 5; i++){
                        TextView textView = new TextView(getContext());
                        textView.setText("a");
                        textView.setTextColor(0xff00ff00);
                        textView.setTextSize(34);
                        //此处以RelativeLayout布局为例，同样LinearLayout也支持该接口
                        RelativeLayout.LayoutParams reLayoutParams =
                                new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
                        reLayoutParams.setMargins(width/2 + ((height / 2 - 25)/ 5 * (2*j - 7)),10 + (height - 50)/ 5 * i,0,0);
                        binding.MainArea.addView(textView, reLayoutParams);
                    }
                }
            }
        });
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.GLView.setEGLContextClientVersion(2);
        DemoRenderer demoRenderer = new DemoRenderer();
        binding.GLView.setRenderer(demoRenderer);
        demoRenderer.setFirstFragment(this);

//
//        handler = new Handler(){
//            @Override
//            public void handleMessage(Message msg) {
//                super.handleMessage(msg);
//                Log.d("FirstFragment","message");
//                if(msg.what == 20020708){
//                    int width = msg.arg1;
//                    int height = msg.arg2;
//                    Log.d("FirstFragment",width + " " + height);
//                    paintText(width, height);
//                }
//
//            }
//        };





//        Log.d("relative layout:",relativeLayout.getLayoutParams().height + " " + relativeLayout.getLayoutParams().width);
//        Log.d("gl view:",binding.GLView.getLayoutParams().height + " " + binding.GLView.getLayoutParams().width);


        // 让text2在父容器中靠右对齐
//        text2LayoutParams.addRule(RelativeLayout.);



//        binding.buttonFirst.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                NavHostFragment.findNavController(FirstFragment.this)
//                        .navigate(R.id.action_FirstFragment_to_SecondFragment);
//            }
//        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

}