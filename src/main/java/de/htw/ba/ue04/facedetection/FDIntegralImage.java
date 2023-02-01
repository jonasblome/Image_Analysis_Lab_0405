package de.htw.ba.ue04.facedetection;

import de.htw.ba.facedetection.IntegralImage;

import java.util.Arrays;

public class FDIntegralImage implements IntegralImage {
	private int[] integral;
	private int width = 0;
	private int height = 0;
	
	public FDIntegralImage(int[] srcPixels, int width, int height) {
		this.width = width;
		this.height = height;
		integral = new int[width * height];
		calculateIntegral(srcPixels);
	}

	private void calculateIntegral(int[] srcPixels) {
		// Loop over every column to calculate column integrals
		for (int x = 0; x < width; x++) {
			int currentColumnSum = 0;

			for (int y = 0; y < height; y++) {
				int pos = y * width + x;
				currentColumnSum += srcPixels[pos] & 0xFF;
				integral[pos] = currentColumnSum;
			}
		}

		// Loop over every column to calculate complete integral
		for (int x = 1; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int pos = y * width + x;

				integral[pos] += integral[pos - 1];
			}
		}
	}
	
	@Override
	public double meanValue(int x, int y, int width, int height) {
		int pos1 = (y + height) * this.width + x + width;
		int pos2 = (y + height) * this.width + x;
		int pos3 = y * this.width + x + width;
		int pos4 = y * this.width + x;

		int areaVal = integral[pos1] - integral[pos2] - integral[pos3] + integral[pos4];

		return areaVal * 1.0 / (width * height);
	}

	@Override
	public void toIntARGB(int[] dstImage) {
		int max = Arrays.stream(integral).max().getAsInt();
		int min = Arrays.stream(integral).min().getAsInt();

		// Normalize, turn to ARGB and copy to destination
		int[] normIntegral = Arrays.stream(integral).map(x -> (int) ((x - min) * 1.0 / (max - min) * 255)).toArray();
		int[] grayIntegral = Arrays.stream(normIntegral).map(x -> 0xFF000000 | x << 16 | x << 8 | x).toArray();
		if (width * height >= 0) System.arraycopy(grayIntegral, 0, dstImage, 0, width * height);
	}

	@Override
	public int getWidth() {
		return width;
	}

	@Override
	public int getHeight() {
		return height;
	}

}
