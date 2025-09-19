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
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

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
                // TODO read onnx external data too
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

        final File[] modelFiles =
                modelDir.listFiles((dir, name) -> name.endsWith(".pte") || name.endsWith(".onnx"));
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
     * Run the provided input through an ONNX model.
     *
     * @param resizedBitmap Input of the expected size for the model, not otherwise preprocessed.
     * @return SegmentationOutput with flattened logits and output shape.
     * @throws OrtException Thrown if an ONNX error occurs.
     */
    private SegmentationOutput runOnnxModel(final Bitmap resizedBitmap) throws OrtException {
        final OrtEnvironment env = OrtEnvironment.getEnvironment();
        final OrtSession session = env.createSession(selectedModelFile.getAbsolutePath(),
                                                     new OrtSession.SessionOptions());

        final OnnxTensor inputTensor = bitmapToOnnxTensor(env, resizedBitmap);
        final String inputName = session.getInputNames().iterator().next();
        final OrtSession.Result result =
                session.run(Collections.singletonMap(inputName, inputTensor));
        final OnnxTensor output = (OnnxTensor) result.get(0);
        final long[] shape = output.getInfo().getShape();
        final float[][][][] logits = (float[][][][]) output.getValue();
        final float[] flatOutput = flatten4D(logits);

        return new SegmentationOutput(flatOutput, shape);
    }

    /**
     * Convert a bitmap to a preprocessed ONNX Tensor.
     *
     * @param env ONNX runtime environment.
     * @param bitmap 3-channeled bitmap to convert.
     * @return Preprocessed ONNX Tensor of shape (1, 3, H, W)
     * @throws OrtException Thrown if there is an ONNX error during tensor creation
     */
    private OnnxTensor bitmapToOnnxTensor(final OrtEnvironment env, final Bitmap bitmap)
            throws OrtException {
        final int height = bitmap.getHeight();
        final int width = bitmap.getWidth();
        final float[] inMean = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
        final float[] inStd = TensorImageUtils.TORCHVISION_NORM_STD_RGB;

        final int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        final FloatBuffer buffer = FloatBuffer.allocate(3 * height * width);

        for (int c = 0; c < 3; c++) {
            for (final int pixel : pixels) {
                final int r = (pixel >> 16) & 0xFF;
                final int g = (pixel >> 8) & 0xFF;
                final int b = pixel & 0xFF;

                final float val;
                if (c == 0) val = (r / 255.0f - inMean[0]) / inStd[0];
                else if (c == 1) val = (g / 255.0f - inMean[1]) / inStd[1];
                else val = (b / 255.0f - inMean[2]) / inStd[2];

                buffer.put(val);
            }
        }

        buffer.rewind();
        final long[] shape = {1, 3, height, width};
        return OnnxTensor.createTensor(env, buffer, shape);
    }

    /**
     * Flatten a 4-dimensional float array to a single dimension.
     *
     * @param inputArray 4-dimensional float array to flatten.
     * @return float array.
     */
    private float[] flatten4D(final float[][][][] inputArray) {
        final int N = inputArray.length;
        final int C = inputArray[0].length;
        final int H = inputArray[0][0].length;
        final int W = inputArray[0][0][0].length;

        final float[] flatArray = new float[N * C * H * W];
        int idx = 0;

        for (final float[][][] blockN : inputArray)
            for (final float[][] blockC : blockN)
                for (final float[] blockH : blockC)
                    for (final float val : blockH) {
                        flatArray[idx] = val;
                        idx++;
                    }

        return flatArray;
    }

    /**
     * Run the provided input through an ExecuTorch model.
     *
     * @param resizedBitmap Input of the expected size for the model, not otherwise preprocessed.
     * @return SegmentationOutput with flattened logits and output shape.
     */
    private SegmentationOutput runTorchModel(final Bitmap resizedBitmap) {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        final Module module = Module.load(selectedModelFile.getAbsolutePath());
        final Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();

        final float[] flatOutput = outputTensor.getDataAsFloatArray();
        final long[] shape = outputTensor.shape();

        return new SegmentationOutput(flatOutput, shape);
    }

    /**
     * Run segmentation inference with the set model and image.
     */
    private void runSegmentation() {
        try {
            // Current test models expect a specific image size and preprocessing
            final int width = 224;
            final int height = 224;

            final Bitmap resizedBitmap =
                    Bitmap.createScaledBitmap(inputBitmap, width, height, true);

            final long timeOne = System.nanoTime();
            final SegmentationOutput segOutput;
            if (selectedModelFile.getAbsolutePath().endsWith(".pte")) {
                segOutput = runTorchModel(resizedBitmap);
            } else {
                segOutput = runOnnxModel(resizedBitmap);
            }

            final long timeTwo = System.nanoTime();
            final Bitmap overlayBitmap = postprocess(segOutput, resizedBitmap);

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
     * @param segOutput inference output with flattened logits and original shape.
     * @param resizedBitmap input image resized to match the model output.
     * @return inference segmentation map with a distinct color for each found class.
     */
    private Bitmap postprocess(final SegmentationOutput segOutput, final Bitmap resizedBitmap) {
        // Deduce class number from the output shape
        final long nClasses = segOutput.shape[1];
        final List<Integer> colors = generateDistinctColors(nClasses);

        final int width = resizedBitmap.getWidth();
        final int height = resizedBitmap.getHeight();

        // For each pixel, output the class (color) with the highest class score
        final float[] scores = segOutput.logits;
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
