package com.example.executorchdemo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_PERMISSIONS = 1;

    private Spinner modelSpinner;
    private TextView modelMemoryText, inferenceTimeText;
    private ImageView inputImageView, outputImageView;
    private Button selectImageButton, runInferenceButton;

    private File selectedModelFile;
    private Bitmap inputBitmap;
    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(), result -> {
                        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                            Uri imageUri = result.getData().getData();
                            try {
                                inputBitmap = MediaStore.Images.Media.getBitmap(
                                        this.getContentResolver(), imageUri);
                                inputImageView.setImageBitmap(inputBitmap);
                            } catch (Exception e) {
                                e.printStackTrace();
                                Toast.makeText(this, "Failed to load image",
                                               Toast.LENGTH_SHORT).show();
                            }
                        }
                    });
    private String modelDirPath;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
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
            public void onItemSelected(
                    final AdapterView<?> parent,
                    final View view,
                    final int pos,
                    final long id
            ) {
                final String fileName = (String) parent.getItemAtPosition(pos);
                selectedModelFile = new File(modelDirPath, fileName);
                final long sizeInKB = selectedModelFile.length() / 1024;
                modelMemoryText.setText("Memory Usage: " + sizeInKB + " KB");
            }

            @Override
            public void onNothingSelected(final AdapterView<?> parent) {}
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
        final File modelDir = new File(getExternalFilesDir(null), "models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        modelDirPath = modelDir.getPath();

        final File[] modelFiles = modelDir.listFiles((dir, name) -> name.endsWith(".pte"));
        final List<String> modelNames = new ArrayList<>();
        if (modelFiles != null) {
            for (final File file : modelFiles) {
                modelNames.add(file.getName());
            }
        }

        if (modelNames.isEmpty()) {
            modelNames.add("No models found");
        }

        final ArrayAdapter<String> adapter =
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, modelNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);

        if (!"No models found".equals(modelNames.get(0))) {
            selectedModelFile = new File(modelDirPath, modelNames.get(0));
        }
    }

    private void selectImage() {
        final Intent pickIntent =
                new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        imagePickerLauncher.launch(pickIntent);
    }

    private void runSegmentation() {
        try {

            final Module module = Module.load(selectedModelFile.getAbsolutePath());

            final int width = 224;
            final int height = 224;

            final Bitmap resizedBitmap =
                    Bitmap.createScaledBitmap(inputBitmap, width, height, true);

            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    resizedBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            final long timeOne = System.nanoTime();
            final Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();
            final long timeTwo = System.nanoTime();

            final long nClasses = outputTensor.shape()[1];
            final List<Integer> colors = generateDistinctColors(nClasses);
            final float[] scores = outputTensor.getDataAsFloatArray();

            final int[] intValues = new int[width * height];
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    int maxi = 0, maxj = 0, maxk = 0;
                    double maxnum = -Double.MAX_VALUE;
                    for (int i = 0; i < nClasses; i++) {
                        final float score = scores[i * (width * height) + j * width + k];
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
            final double inferenceTimeMs = (timeTwo - timeOne) / 1_000_000.0;
            final double arrayTimeMs = (timeThree - timeTwo) / 1_000_000.0;

            final Bitmap overlayBitmap =
                    Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            overlayBitmap.setPixels(intValues, 0, width, 0, 0, width, height);

            final Bitmap outputBitmap = overlayWithAlpha(resizedBitmap, overlayBitmap, 0.5f);
            final Bitmap finalBitmap =
                    Bitmap.createScaledBitmap(outputBitmap, inputBitmap.getWidth(),
                                              inputBitmap.getHeight(), true);

            outputImageView.setImageBitmap(finalBitmap);
            inferenceTimeText.setText(
                    "Inference Time: " + String.format("%.2f", inferenceTimeMs) + " ms");
            Log.d("ImageSegmentation", "inference time (ms): " + inferenceTimeMs);
            Log.d("ImageSegmentation", "array time (ms): " + arrayTimeMs);

        } catch (final Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Inference failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private List<Integer> generateDistinctColors(final long count) {
        final List<Integer> colors = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            final float hue = (i * 360f / count) % 360f;
            final float[] hsv = {hue, 0.7f, 1.0f};
            colors.add(Color.HSVToColor(hsv));
        }
        return colors;
    }

    private Bitmap overlayWithAlpha(final Bitmap base, final Bitmap overlay, final float alpha) {
        if (base.getWidth() != overlay.getWidth() || base.getHeight() != overlay.getHeight()) {
            throw new IllegalArgumentException("Bitmaps must be the same size");
        }

        final Bitmap result = base.copy(base.getConfig(), true);
        final Canvas canvas = new Canvas(result);

        final Paint paint = new Paint();
        paint.setAlpha((int) (alpha * 255));
        canvas.drawBitmap(overlay, 0, 0, paint);

        return result;
    }

    private void requestPermissionsIfNeeded() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_PERMISSIONS);
        }
    }

}
