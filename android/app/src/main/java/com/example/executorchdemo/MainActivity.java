package com.example.executorchdemo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
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

            Module module = Module.load(selectedModelFile.getAbsolutePath());

            int width = 224;
            int height = 224;

            Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, width, height, true);
            FloatBuffer inputBuffer = Tensor.allocateFloatBuffer(3 * width * height);

            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            final long timeOne = System.nanoTime();
            Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();
            final long timeTwo = System.nanoTime();

            final long nClasses = outputTensor.shape()[1];
            List<Integer> colors = generateDistinctColors(nClasses);
            final float[] scores = outputTensor.getDataAsFloatArray();

            int[] intValues = new int[width * height];
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    int maxi = 0, maxj = 0, maxk = 0;
                    double maxnum = -Double.MAX_VALUE;
                    for (int i = 0; i < nClasses; i++) {
                        float score = scores[i * (width * height) + j * width + k];
                        if (score > maxnum) {
                            maxnum = score;
                            maxi = i;
                            maxj = j;
                            maxk = k;
                        }
                    }
                    if (maxi == 0) intValues[maxj * width + maxk] = resizedBitmap.getPixel(k, j);
                    else intValues[maxj * width + maxk] = colors.get(maxi);
                }
            }
            final long timeThree = System.nanoTime();
            double inferenceTimeMs = (timeTwo - timeOne) / 1_000_000.0;
            double arrayTimeMs = (timeThree - timeTwo) / 1_000_000.0;

            Bitmap overlayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            overlayBitmap.setPixels(intValues, 0, width, 0, 0, width, height);

            Bitmap outputBitmap = overlayWithAlpha(resizedBitmap, overlayBitmap, 0.5f);
            Bitmap finalBitmap = Bitmap.createScaledBitmap(outputBitmap, inputBitmap.getWidth(),
                    inputBitmap.getHeight(), true);

            outputImageView.setImageBitmap(finalBitmap);
            inferenceTimeText.setText("Inference Time: " + String.format("%.2f", inferenceTimeMs) + " ms");
            Log.d("ImageSegmentation", "inference time (ms): " + inferenceTimeMs);
            Log.d("ImageSegmentation", "array time (ms): " + arrayTimeMs);

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Inference failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private List<Integer> generateDistinctColors(long count) {
        List<Integer> colors = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            float hue = (i * 360f / count) % 360f;
            float[] hsv = new float[]{hue, 0.7f, 1.0f};
            colors.add(Color.HSVToColor(hsv));
        }
        return colors;
    }

    private Bitmap overlayWithAlpha(Bitmap base, Bitmap overlay, float alpha) {
        if (base.getWidth() != overlay.getWidth() || base.getHeight() != overlay.getHeight()) {
            throw new IllegalArgumentException("Bitmaps must be the same size");
        }

        Bitmap result = base.copy(base.getConfig(), true);
        Canvas canvas = new Canvas(result);

        Paint paint = new Paint();
        paint.setAlpha((int) (alpha * 255));
        canvas.drawBitmap(overlay, 0, 0, paint);

        return result;
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
