package com.example.executorchdemo;

public class SegmentationOutput {
    public final float[] logits;
    public final long[] shape;

    public SegmentationOutput(final float[] logits, final long[] shape) {
        this.logits = logits;
        this.shape = shape;
    }
}
