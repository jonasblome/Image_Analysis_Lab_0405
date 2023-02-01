/**
 * @author Nico Hezel
 */
package de.htw.ba.ue04;

import java.io.IOException;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class BA_Ue05_Jonas_Blome {

	public static void main(String[] args) throws IOException {
		Application.launch(FXApplication.class);
	}

	/**
	 * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
	 * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
	 * 
	 * @author Nico Hezel
	 *
	 */
	public static class FXApplication extends Application {

		@Override
		public void start(Stage stage) throws Exception {

			Parent ui = new FXMLLoader(getClass().getResource("/de/htw/ba/ue04/view/FaceDetectionView.fxml")).load();
			Scene scene = new Scene(ui);
			stage.setScene(scene);
			stage.setTitle("Face Detection - Jonas Blome");
			stage.show();
		}
	}
}