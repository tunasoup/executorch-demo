package com.example.executorchdemo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.*;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_PERMISSIONS = 1;

    private Spinner modelSpinner;
    private TextView modelMemoryText, inferenceTimeText;
    private ImageView inputImageView, outputImageView;
    private Button selectImageButton, runInferenceButton;

    private File selectedModelFile;
    private Bitmap inputBitmap;

    private String modelDirPath;

    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        inputBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                        inputImageView.setImageBitmap(inputBitmap);
                    } catch (Exception e) {
                        e.printStackTrace();
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                    }
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestPermissionsIfNeeded();

        modelSpinner = findViewById(R.id.modelSpinner);
        modelMemoryText = findViewById(R.id.modelMemoryText);
        inferenceTimeText = findViewById(R.id.inferenceTimeText);
        inputImageView = findViewById(R.id.inputImageView);
        outputImageView = findViewById(R.id.outputImageView);
        selectImageButton = findViewById(R.id.selectImageButton);
        runInferenceButton = findViewById(R.id.runInferenceButton);

        loadModelList();

        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                String fileName = (String) parent.getItemAtPosition(pos);
                selectedModelFile = new File(modelDirPath, fileName);
                long sizeInKB = selectedModelFile.length() / 1024;
                modelMemoryText.setText("Memory Usage: " + sizeInKB + " KB");
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        selectImageButton.setOnClickListener(v -> selectImage());

        runInferenceButton.setOnClickListener(v -> {
            if (inputBitmap == null || selectedModelFile == null) {
                Toast.makeText(this, "Please select a model and image", Toast.LENGTH_SHORT).show();
                return;
            }
            runSegmentation();
        });
    }

    private void loadModelList() {
        File modelDir = new File(getExternalFilesDir(null), "models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        modelDirPath = modelDir.getPath();

        File[] modelFiles = modelDir.listFiles((dir, name) -> name.endsWith(".pte"));
        List<String> modelNames = new ArrayList<>();
        if (modelFiles != null) {
            for (File file : modelFiles) {
                modelNames.add(file.getName());
            }
        }

        if (modelNames.isEmpty()) {
            modelNames.add("No models found");
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, modelNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);

        if (!modelNames.get(0).equals("No models found")) {
            selectedModelFile = new File(modelDirPath, modelNames.get(0));
        }
    }

    private void selectImage() {
        Intent pickIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        imagePickerLauncher.launch(pickIntent);
    }

    private void runSegmentation() {
        try {
            long startTime = System.nanoTime();

            Module module = Module.load(selectedModelFile.getAbsolutePath());

            Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, 224, 224, true);
            FloatBuffer inputBuffer = Tensor.allocateFloatBuffer(3 * 224 * 224);

            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();
            float[] outputArray = outputTensor.getDataAsFloatArray();

            // For demo: use outputArray as grayscale mask and render
            Bitmap outputBitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
            for (int y = 0; y < 224; y++) {
                for (int x = 0; x < 224; x++) {
                    int index = y * 224 + x;
                    int intensity = (int) (outputArray[index] * 255);
                    int pixel = 0xFF000000 | (intensity << 16) | (intensity << 8) | intensity;
                    outputBitmap.setPixel(x, y, pixel);
                }
            }

            long endTime = System.nanoTime();
            double inferenceTimeMs = (endTime - startTime) / 1_000_000.0;

            outputImageView.setImageBitmap(outputBitmap);
            inferenceTimeText.setText("Inference Time: " + String.format("%.2f", inferenceTimeMs) + " ms");

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Inference failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void requestPermissionsIfNeeded() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_PERMISSIONS);
        }
    }

}
