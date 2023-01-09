package com.example.mindgraph;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity {

    TextView textView;

    EditText inputText;

    Button button;

    String textInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = (TextView)findViewById(R.id.textview);
        inputText = (EditText) findViewById(R.id.inputText);
    }

    /*CODE WHEN THE BUTTON IS CLICKED*/
    public void updateText(View view){
        textInput = String.valueOf(inputText.getText());

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        //THIS WILL START PYTHON

        //now create python instance
        Python py = Python.getInstance();

        //create python object
        //PyObject pyobj = py.getModule("myscript"); //give python script name
        //PyObject pyobj = py.getModule("try_sentimentAnalysis");
        PyObject pyobj = py.getModule("less_runtime");

        //call the function of the python file
        PyObject obj = pyobj.callAttr("output", textInput);
        //PyObject obj = pyobj.callAttr("output");

        //this set returned text to the textview activity
        textView.setText(obj.toString());

        System.out.println("button Clicked");
    }
}