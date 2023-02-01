/**
 * @author Nico Hezel, Klaus Jung
 */
package de.htw.ba.ue04.controller;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;

import de.htw.ba.facedetection.Evaluation;
import de.htw.ba.facedetection.IntegralImage;
import de.htw.ba.facedetection.StrongClassifier;
import de.htw.ba.facedetection.TestImage;
import de.htw.ba.facedetection.TestImage.Subject;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

public abstract class FaceDetectionBase {

	private enum LeftViewMethode { PlainImage, TrainingSet, StrongClassifier, DetectionResult };
	
	private enum RightViewMethode { GrayscaleImage, IntegralImage, FeatureHeatMap };
	
	@FXML
	private ImageView leftImageView;

	@FXML
	private ImageView rightImageView;

	@FXML
	private Label runtimeLabel;

	@FXML
	private Label thresholdValue;

	@FXML
	private Slider thresholdSlider;
	
	@FXML
	private CheckBox nonMaxSuppressionCheckBox;

	@FXML
	private Slider weakClassifierCountSlider;

	@FXML
	private Slider trainRegionCountSlider;

	@FXML
	private ComboBox<Subject> subjectSelection;
	
	@FXML
	private ComboBox<LeftViewMethode> leftViewSelection;

	@FXML
	private ComboBox<RightViewMethode> rightViewSelection;
	
	@FXML
	private RadioButton manualButton;
	
	@FXML
	private RadioButton adaBoostButton;
	
	@FXML
	private CheckBox demoModeCheckBox;
	
	protected TestImage testImage = null;
	protected IntegralImage integralImage = null;
	protected StrongClassifier strongClassifier = null;
	protected List<Rectangle> detectionResult = null;
	
	protected int[] featureMapPixels = null;
	
	protected boolean isDemo = false;
	
	@FXML
	public void initialize() {
		
		// TODO: For faster debugging you may change the default image here.
		subjectSelection.getItems().addAll(Subject.values());
		subjectSelection.setValue(Subject.PERSON); // PERSON
		subjectSelection.setOnAction(this::loadImage);		
		
		// TODO: For faster debugging you may change the default left view here.
		leftViewSelection.getItems().addAll(LeftViewMethode.values());
		leftViewSelection.setValue(LeftViewMethode.PlainImage); // PlainImage
		leftViewSelection.setOnAction(this::runMethod);
		
		// TODO: For faster debugging you may change the default right view here.
		rightViewSelection.getItems().addAll(RightViewMethode.values());
		rightViewSelection.setValue(RightViewMethode.GrayscaleImage); // GrayscaleImage
		rightViewSelection.setOnAction(this::runMethod);
		
		// TODO: For faster debugging you may change the default value here.
		nonMaxSuppressionCheckBox.setSelected(false); // false
		
		// Update the label when the slider is changing.
		thresholdSlider.valueProperty().addListener((ov, oldVal, newVal) -> {
			thresholdValue.setText(String.format("%.2f", newVal));
		});
		
		// Trigger new calculation when slider stopped changing.
		thresholdSlider.valueChangingProperty().addListener((ov, oldVal, newVal) -> {
			if (newVal == false) {
				detectionResult = null;
				runMethod(null);
			}
		});

		weakClassifierCountSlider.valueChangingProperty().addListener(this::weakClassifierCountChange);
		trainRegionCountSlider.valueChangingProperty().addListener(this::trainRegionCountChange);
		
		// load the default image.
		loadImage(null);
	}


	/**
	 * Called when the training regions slider is changed.
	 * To be used in Exercise 5 only.
	 * 
	 * @param ov
	 * @param old_val
	 * @param new_val
	 */
	protected void trainRegionCountChange(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {
		if (new_val == false) {

			// Create more non-face regions for training and evaluation,
			testImage.addNonFaceRegions(trainRegionCountSlider.valueProperty().intValue());
			
			if (adaBoostButton.isSelected()) {
				// Trigger a new training.
				strongClassifier = null;
			}
			// Trigger new detection and evaluation.
			detectionResult = null;
			runMethod(null);
		}
	}
	
	/**
	 * Called when the number of weak classifiers is changed.
	 * To be used in Exercise 5 only.
	 * 
	 * @param ov
	 * @param old_val
	 * @param new_val
	 */
	protected void weakClassifierCountChange(ObservableValue<? extends Boolean> ov, Boolean old_val, Boolean new_val) {
		if (new_val == false) {
			
			if (adaBoostButton.isSelected()) {
				// Trigger a new training with the adjusted number of weak classifiers.
				strongClassifier = null;
				detectionResult = null;
			}
			runMethod(null);
		}
	}
	
	/**
	 * Load a new test image, depending on the users choice in the subjectSelection combo box.
	 * 
	 * @param event
	 */
	@FXML
	public void loadImage(ActionEvent event) {

		try {
			
			// Load test image.
			Subject subject = subjectSelection.getSelectionModel().getSelectedItem();
			testImage = TestImage.createTestImage(subject);
			testImage.addNonFaceRegions(trainRegionCountSlider.valueProperty().intValue());

			// Draw the test image on the left hand side.
			leftImageView.setImage(pixelToImage(testImage.getPixels(), testImage.getWidth(), testImage.getHeight()));
			
			integralImage = null;
			strongClassifier = null;
			detectionResult = null;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		runMethod(null);
	}

	/**
	 * Manual or AdaBoost mode has changed.
	 * 
	 * @param event
	 */
	@FXML
	public void modeSelection(ActionEvent event) {
		demoModeCheckBox.setDisable(!manualButton.isSelected());
		if (strongClassifier != null) {
			// Trigger new calculations.
			strongClassifier = null;
			detectionResult = null;
			runMethod(null);
		}
	}
	
	/**
	 * Non-maximum suppression has been switched.
	 * 
	 * @param event
	 */
	@FXML
	public void nonMaxSuppressionChanged(ActionEvent event) {
		// Trigger new detection.
		detectionResult = null;
		runMethod(null);	
	}
	
	/**
	 * Demo mode has been altered.
	 * 
	 * @param event
	 */
	@FXML
	public void demoModeChanged(ActionEvent event) {
		// Trigger new calculations.
		isDemo = demoModeCheckBox.isSelected();
		strongClassifier = null;
		detectionResult = null;
		runMethod(null);	
	}
	
	/**
	 * Create the integral image out of the given source pixels.
	 * 
	 * @param srcPixels
	 */
	protected abstract void calculateIntegralImage(int[] srcPixels, int width, int height);

	/**
	 * Create a strong classifier out of some manually created weak classifiers.
	 */
	protected abstract void createManualClassifier();

	/**
	 * Randomly create the given number of weak classifiers, train them with
	 * AdaBoost, and combine them to a strong classifier.
	 * 
	 * @param weakClassifierCount
	 */
	protected abstract void createTrainedClassifier(int weakClassifierCount);
	
	/**
	 * Called when new calculations are needed. Depending on missing
	 * components, they are (re)calculated. Depending on current user
	 * selections results are displayed in left and right view.
	 * 
	 * @param event
	 */
	@FXML
	public void runMethod(ActionEvent event) {

		// no images loaded
		if(leftImageView.getImage() == null)
			return;

		// get image dimensions
		int width = testImage.getWidth();
		int height = testImage.getHeight();

		// get pixels arrays
		int srcPixels[] = Arrays.copyOf(testImage.getPixels(), testImage.getPixels().length);
		int dstPixels[] = new int[width * height];
		
		long startTime = System.currentTimeMillis();

		// create integral image
		if(integralImage == null) {
			calculateIntegralImage(srcPixels, width, height);
		}
		
		// create strong classifier
		if(strongClassifier == null) {
			if(manualButton.isSelected()) {
				createManualClassifier();				
			} else {
				createTrainedClassifier(weakClassifierCountSlider.valueProperty().intValue());				
			}
		}
		
		// get viewing choice 
		LeftViewMethode leftViewMethod = leftViewSelection.getSelectionModel().getSelectedItem();
		RightViewMethode rightViewMethod = rightViewSelection.getSelectionModel().getSelectedItem();

		// detect image
		if(detectionResult == null &&
		   (leftViewMethod == LeftViewMethode.DetectionResult || 
			rightViewMethod == RightViewMethode.FeatureHeatMap)) {
			
			float threshold = (float)thresholdSlider.getValue();
			featureMapPixels = new int[width * height];
			
			doDetection(featureMapPixels, width, height, threshold, nonMaxSuppressionCheckBox.isSelected());
	     	
			// evaluates the detection result
	     	Evaluation.evaluate(detectionResult, testImage);
		}

		// draw left image
		switch (leftViewMethod) {
		case TrainingSet:
			drawTrainingSet(srcPixels, width, height);
			break;
		case StrongClassifier:
			drawStrongClassifier(srcPixels, width, height);
			break;
		case DetectionResult:
			drawDetectionResult(srcPixels, width, height);
			break;
		default:
			break;
		}
		
		// draw right image
		switch (rightViewMethod) {
		case GrayscaleImage: // zeichne das Graustufenbild
			doGray(testImage.getPixels(), width, height, dstPixels, width, height);
			break;
		case IntegralImage:	// zeichne das Integral-Bild
			integralImage.toIntARGB(dstPixels);
			break;
		case FeatureHeatMap:
			if (featureMapPixels != null)
				System.arraycopy(featureMapPixels, 0, dstPixels, 0, featureMapPixels.length);
			break;
		default:
			break;
		}

		rightImageView.setImage(pixelToImage(dstPixels, width, height));
		leftImageView.setImage(pixelToImage(srcPixels, width, height));
		runtimeLabel.setText("Calculations took " + (System.currentTimeMillis() - startTime) + " ms");
	}

	/**
     * Uses strongClassifier to calculate a feature heat map.
     * Stores all detected regions in the detectionResult list of rectangles.
	 * 
	 * @param featureHeatMapPixels
	 * @param width
	 * @param height
	 * @param threshold
	 * @param nonMaxSuppression
	 */
	protected abstract void doDetection(int[] featureHeatMapPixels, int width, int height, float threshold, boolean nonMaxSuppression);

	// ------------------------------------------------------------------------------------
	// --------------------------------------- Helpers ------------------------------------
	// ------------------------------------------------------------------------------------
	
	public static void doGray(int srcPixels[], int srcWidth, int srcHeight, int dstPixels[], int dstWidth, int dstHeight) {
		for (int y = 0; y < dstHeight; y++) {
			for (int x = 0; x < dstWidth; x++) {
				int pos	= y * dstWidth + x;
				
				int c = srcPixels[y * srcWidth + x]; // RGB Value
				int r = (c>>16)&0xFF;
				int g = (c>> 8)&0xFF;
				int b = (c    )&0xFF;
					
				int lum = (int) (0.299*r + 0.587*g + 0.114*b); // Grauwert
				lum = Math.min(lum,255);
				dstPixels[pos] = 0xFF000000 | (lum<<16) | (lum<<8) | lum;
			}
		}
    }
	
	public void drawTrainingSet(int srcPixels[], int srcWidth, int srcHeight) {
		//System.out.println("drawTrainingSet");
		if (testImage == null)
			return;		
		
     	// copy srcPixels into new image
		BufferedImage bufferedImage = new BufferedImage(srcWidth, srcHeight, BufferedImage.TYPE_INT_ARGB);
    	bufferedImage.setRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth);
    	Graphics2D g2d = bufferedImage.createGraphics();
    	
    	// draw all face and non-face regions of the test image
    	g2d.setColor(Color.RED);
    	for (Rectangle rect : testImage.getNonFaceRectangles())
    		g2d.drawRect((int)rect.getX(), (int)rect.getY(), (int)rect.getWidth(), (int)rect.getHeight());
    	
    	g2d.setColor(Color.GREEN);
    	for (Rectangle rect : testImage.getFaceRectangles())
    		g2d.drawRect((int)rect.getX(), (int)rect.getY(), (int)rect.getWidth(), (int)rect.getHeight());
			
     	// write back image's pixels to srcPixels
    	g2d.dispose();
		bufferedImage.getRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth);
	}
    
	public void drawStrongClassifier(int srcPixels[], int srcWidth, int srcHeight) {
		//System.out.println("drawStrongClassifier");
		if (strongClassifier == null)
			return;
		
     	// copy srcPixels into new image
		BufferedImage bufferedImage = new BufferedImage(srcWidth, srcHeight, BufferedImage.TYPE_INT_ARGB);
    	bufferedImage.setRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth);
    	Graphics2D g2d = bufferedImage.createGraphics();
    	
    	// draw the strongClassifier's layout into each face region of the test image
    	for(Rectangle faceRect : testImage.getFaceRectangles()) {
        	strongClassifier.drawAt(g2d, faceRect.x, faceRect.y);
    	}
    	
     	// write back image's pixels to srcPixels
    	g2d.dispose();
		bufferedImage.getRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth); 	
	}
    
	public void drawDetectionResult(int srcPixels[], int srcWidth, int srcHeight) {
		//System.out.println("drawDetectionResult");
		if (strongClassifier == null)
			return;
		
     	// copy srcPixels into new image
		BufferedImage bufferedImage = new BufferedImage(srcWidth, srcHeight, BufferedImage.TYPE_INT_ARGB);
    	bufferedImage.setRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth);
    	Graphics2D g2d = bufferedImage.createGraphics();
    	
    	// Draw all detected matches. Use the evaluation result to mark 
    	// true positive and false positive by green and red color respectively.
    	for (Rectangle predictedFaceRect : detectionResult) {
			g2d.setColor((Evaluation.isFace(predictedFaceRect, testImage)) ? Color.GREEN : Color.RED);
			g2d.drawRect(predictedFaceRect.x, predictedFaceRect.y, predictedFaceRect.width, predictedFaceRect.height);
    	}
    	
     	// write back image's pixels to srcPixels
    	g2d.dispose();
		bufferedImage.getRGB(0, 0, srcWidth, srcHeight, srcPixels, 0, srcWidth);    	
	}
    
	public static int[] imageToPixel(Image image) {
		int width = (int) image.getWidth();
		int height = (int) image.getHeight();
		int[] pixels = new int[width * height];
		image.getPixelReader().getPixels(0, 0, width, height, PixelFormat.getIntArgbInstance(), pixels, 0, width);
		return pixels;
	}

	public static int[] imageToPixel(BufferedImage image) {
		int width = (int) image.getWidth();
		int height = (int) image.getHeight();
		int[] pixels = new int[width * height];
		image.getRGB(0, 0, width, height, pixels, 0, width);
		return pixels;
	}

	public static Image pixelToImage(int[] pixels, int width, int height) {
		WritableImage wr = new WritableImage(width, height);
		PixelWriter pw = wr.getPixelWriter();
		pw.setPixels(0, 0, width, height, PixelFormat.getIntArgbInstance(), pixels, 0, width);
		return wr;
	}
}
