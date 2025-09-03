package com.example.executorchdemo;

import android.content.Intent;
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
import androidx.annotation.StringRes;
import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private Spinner modelSpinner;
    private TextView modelMemoryView, inferenceTimeView, postprocessingTimeView;
    private ImageView inputImageView, outputImageView;

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

        modelSpinner = findViewById(R.id.modelSpinner);
        modelMemoryView = findViewById(R.id.modelMemoryText);
        inferenceTimeView = findViewById(R.id.inferenceTimeText);
        postprocessingTimeView = findViewById(R.id.postprocessingTimeText);
        inputImageView = findViewById(R.id.inputImageView);
        outputImageView = findViewById(R.id.outputImageView);
        final Button selectImageButton = findViewById(R.id.selectImageButton);
        final Button runInferenceButton = findViewById(R.id.runInferenceButton);

        resetInferenceTimes();

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
                setViewText(R.string.memory_usage, modelMemoryView, sizeInKB + "KB");
            }

            @Override
            public void onNothingSelected(final AdapterView<?> parent) {
                setViewText(R.string.memory_usage, modelMemoryView, "0KB");
            }

        });

        selectImageButton.setOnClickListener(v -> selectImage());

        runInferenceButton.setOnClickListener(v -> {
            if (inputBitmap == null || selectedModelFile == null) {
                Toast.makeText(this, getString(R.string.selections_required),
                               Toast.LENGTH_SHORT).show();
                return;
            }
            runSegmentation();
        });
    }

    /**
     * Set a TextView's parameter section to the given text.
     *
     * @param resId string resource id with a single parameter string format
     * @param textView TextView UI element
     * @param text string as the resource's argument
     */
    private void setViewText(
            @StringRes final int resId, final TextView textView,
            final String text
    ) {
        textView.setText(getString(resId, text));
    }

    /**
     * Load the inference models from the model directory, and place them on the spinner.
     */
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

        final String noModels = getString(R.string.no_models_found);

        if (modelNames.isEmpty()) {
            modelNames.add(noModels);
        }

        final ArrayAdapter<String> adapter =
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, modelNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);

        if (!noModels.equals(modelNames.get(0))) {
            selectedModelFile = new File(modelDirPath, modelNames.get(0));
        }
    }

    private void selectImage() {
        final Intent pickIntent =
                new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        imagePickerLauncher.launch(pickIntent);
    }

    /**
     * Run segmentation inference with the set model and image.
     */
    private void runSegmentation() {
        try {
            final Module module = Module.load(selectedModelFile.getAbsolutePath());

            // Current test model expects a specific image size and preprocessing
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
            final Bitmap overlayBitmap = postprocess(outputTensor, resizedBitmap);

            final Bitmap outputBitmap = overlayWithAlpha(resizedBitmap, overlayBitmap, 0.5f);
            final Bitmap finalBitmap =
                    Bitmap.createScaledBitmap(outputBitmap, inputBitmap.getWidth(),
                                              inputBitmap.getHeight(), true);

            final long timeThree = System.nanoTime();
            final double inferenceTimeMs = (timeTwo - timeOne) / 1_000_000.0;
            final double postprocessTimeMs = (timeThree - timeTwo) / 1_000_000.0;

            outputImageView.setImageBitmap(finalBitmap);
            final String inferenceTime = String.format("%.2f", inferenceTimeMs) + " ms";
            final String arrayTime = String.format("%.2f", postprocessTimeMs) + " ms";
            setViewText(R.string.inference_time, inferenceTimeView, inferenceTime);
            setViewText(R.string.postprocessing_time, postprocessingTimeView, arrayTime);
            Log.d("ImageSegmentation", "inference time (ms): " + inferenceTimeMs);
            Log.d("ImageSegmentation", "array time (ms): " + postprocessTimeMs);

        } catch (final Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Inference failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Convert the model output into a readable segmentation image.
     *
     * @param outputTensor inference output of shape (images, classes, height, width).
     * @param resizedBitmap input image resized to match the model output.
     * @return inference segmentation map with a distinct color for each found class.
     */
    private Bitmap postprocess(final Tensor outputTensor, final Bitmap resizedBitmap) {
        // Deduce class number from the output shape
        final long nClasses = outputTensor.shape()[1];
        final List<Integer> colors = generateDistinctColors(nClasses);

        final int width = resizedBitmap.getWidth();
        final int height = resizedBitmap.getHeight();

        // For each pixel, output the class (color) with the highest class score
        final float[] scores = outputTensor.getDataAsFloatArray();
        final int[] intValues = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int bestClassIdx = 0;
                double bestScore = -Double.MAX_VALUE;
                for (int c = 0; c < nClasses; c++) {
                    // Get the corresponding pixel on the flattened array
                    final float score = scores[c * (width * height) + y * width + x];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClassIdx = c;
                    }
                }
                // Keep the original pixel color for background class (index 0)
                if (bestClassIdx == 0) intValues[y * width + x] = resizedBitmap.getPixel(x, y);
                else intValues[y * width + x] = colors.get(bestClassIdx);
            }
        }
        final Bitmap segmentationMap =
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        segmentationMap.setPixels(intValues, 0, width, 0, 0, width, height);
        return segmentationMap;
    }

    private void resetInferenceTimes() {
        setViewText(R.string.inference_time, inferenceTimeView, "-");
        setViewText(R.string.postprocessing_time, postprocessingTimeView, "-");
    }

    /**
     * Generate colors with different hues. Meant to be used for differentiating different classes
     * in segmented images.
     *
     * @param count number of colors to generate.
     * @return list of distinct colors.
     */
    private List<Integer> generateDistinctColors(final long count) {
        final List<Integer> colors = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            final float hue = (i * 360f / count) % 360f;
            final float[] hsv = {hue, 0.7f, 1.0f};
            colors.add(Color.HSVToColor(hsv));
        }
        return colors;
    }

    /**
     * Overlay an image on top of another one with a given alpha value.
     *
     * @param base bottom layer image with full opacity.
     * @param overlay image to add on top of the base image.
     * @param alpha opacity of the overlay image between 0 and 1.
     * @return combined image with the overlay image on top of the base image.
     */
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

}
