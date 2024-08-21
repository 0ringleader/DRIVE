using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class UIButtonHandler : MonoBehaviour
{
    public Material groundMaterial; // Reference to the ground material
    public Texture groundTexture; // Reference to the ground texture
    public CustomCarController carController; // Reference to the car controller
    public MJPEGStream mjpegStream; // Reference to the MJPEG stream class
    private TextMeshProUGUI mouseWheelText; // Reference to the TextMeshPro UI element
    private TextMeshProUGUI speedText; // Reference to the TextMeshPro UI element for speed
    private TextMeshProUGUI steeringText; // Reference to the TextMeshPro UI element for steering
    private Slider gameSpeedSlider; // Reference to the game speed slider
    private TextMeshProUGUI gameSpeedText; // Reference to the TextMeshPro UI element for game speed
    private TMP_Dropdown resolutionDropdown; // Reference to the resolution dropdown
    
    void Start()
    {
        // Find the UI elements in the scene
        mouseWheelText = GameObject.Find("MousewheelText").GetComponent<TextMeshProUGUI>();
        speedText = GameObject.Find("SpeedText").GetComponent<TextMeshProUGUI>();
        steeringText = GameObject.Find("SteeringText").GetComponent<TextMeshProUGUI>();
        gameSpeedSlider = GameObject.Find("GameSpeedSlider").GetComponent<Slider>();
        gameSpeedText = GameObject.Find("GameSpeedText").GetComponent<TextMeshProUGUI>();

        Toggle toggleTextureToggle = GameObject.Find("ToggleTextureToggle").GetComponent<Toggle>();
        Button exitButton = GameObject.Find("ExitButton").GetComponent<Button>();
        Button resetCarButton = GameObject.Find("ResetCarButton").GetComponent<Button>();
        Toggle controlByWebsiteToggle = GameObject.Find("ControlByWebsiteToggle").GetComponent<Toggle>();
        Toggle toggleOffRoadReset = GameObject.Find("ToggleOffRoadReset").GetComponent<Toggle>();
        Toggle syncToGameSpeedToggle = GameObject.Find("SyncToGameSpeedToggle").GetComponent<Toggle>();

        // Add listeners to the toggles and buttons
        toggleTextureToggle.onValueChanged.AddListener(delegate { ToggleGroundTexture(toggleTextureToggle); });
        exitButton.onClick.AddListener(ExitApplication);
        resetCarButton.onClick.AddListener(ResetCar);
        controlByWebsiteToggle.onValueChanged.AddListener(delegate { ToggleControlByWebsite(controlByWebsiteToggle); });
        toggleOffRoadReset.onValueChanged.AddListener(delegate { ToggleOffRoadReset(toggleOffRoadReset); });
        resolutionDropdown = GameObject.Find("ResolutionDropdown").GetComponent<TMP_Dropdown>();
        
        // Add listener to the syncToGameSpeed toggle
        syncToGameSpeedToggle.onValueChanged.AddListener(delegate { ToggleSyncToGameSpeed(syncToGameSpeedToggle); });

        // Add listener to the slider
        gameSpeedSlider.onValueChanged.AddListener(UpdateGameSpeed);

        PopulateResolutionDropdown();
        
        // Add listener to the resolution dropdown
        resolutionDropdown.onValueChanged.AddListener(delegate { ChangeResolution(resolutionDropdown); });

        // Initialize the content of the mouse wheel text
        ToggleMouseWheelText(controlByWebsiteToggle.isOn);

        // Set initial game speed
        UpdateGameSpeed(gameSpeedSlider.value);
    }
    
    void PopulateResolutionDropdown()
    {
        resolutionDropdown.ClearOptions(); // Alte Optionen entfernen

        Resolution[] resolutions = Screen.resolutions; // Alle verfügbaren Auflösungen abrufen
        List<string> options = new List<string>();

        foreach (Resolution res in resolutions)
        {
            string option = res.width + " x " + res.height;
            options.Add(option);
        }

        resolutionDropdown.AddOptions(options); // Optionen hinzufügen
        resolutionDropdown.onValueChanged.AddListener(delegate { SetResolution(resolutionDropdown.value); });
    }

    void SetResolution(int resolutionIndex)
    {
        Resolution resolution = Screen.resolutions[resolutionIndex];
        Screen.SetResolution(resolution.width, resolution.height, Screen.fullScreen);
    }

    void ToggleGroundTexture(Toggle toggle)
    {
        if (toggle.isOn)
        {
            groundMaterial.SetTexture("_MainTex", groundTexture);
        }
        else
        {
            groundMaterial.SetTexture("_MainTex", null);
        }
    }

    void ExitApplication()
    {
        Application.Quit();
    }

    void ResetCar()
    {
        if (carController != null)
        {
            carController.ResetCar();
        }
    }

    void ToggleControlByWebsite(Toggle toggle)
    {
        if (carController != null)
        {
            carController.isControlledByWebsite = toggle.isOn;
        }
        ToggleMouseWheelText(toggle.isOn);
    }

    void ToggleMouseWheelText(bool isControlledByWebsite)
    {
        if (mouseWheelText != null)
        {
            mouseWheelText.text = isControlledByWebsite ? "" : "Control speed with mousewheel or arrow keys";
        }
    }

    void ToggleOffRoadReset(Toggle toggle)
    {
        if (carController != null)
        {
            carController.resetOnOffRoad = toggle.isOn;
        }
    }

    // New function to toggle syncToGameSpeed
    void ToggleSyncToGameSpeed(Toggle toggle)
    {
        if (mjpegStream != null)
        {
            mjpegStream.syncToGameSpeed = toggle.isOn;
        }
    }

    public void UpdateSpeedText(float speed, float mappedSpeed)
    {
        if (speedText != null)
        {
            speedText.text = $"Speed: {speed:F2}m/s, value: {mappedSpeed:F0}";
        }
    }

    public void UpdateSteeringText(float steering, float mappedSteering)
    {
        if (steeringText != null)
        {
            steeringText.text = $"Steering: {steering:F2}°, value: {mappedSteering:F0}";
        }
    }

    void UpdateGameSpeed(float newSpeed)
    {
        newSpeed = newSpeed / 10f;
        Time.timeScale = newSpeed;

        // Optional: Update the UI text with the current game speed
        if (gameSpeedText != null)
        {
            gameSpeedText.text = $"Game Speed: {newSpeed:F2}x";
        }
        mjpegStream.setGameSpeed(newSpeed);
    }
    void ChangeResolution(TMP_Dropdown dropdown)
    {
        // Change the screen resolution based on the dropdown selection
        switch (dropdown.value)
        {
            case 0:
                Screen.SetResolution(640, 480, Screen.fullScreen); // Option 1: 640x480
                break;
            case 1:
                Screen.SetResolution(1280, 720, Screen.fullScreen); // Option 2: 1280x720
                break;
            case 2:
                Screen.SetResolution(1920, 1080, Screen.fullScreen); // Option 3: 1920x1080
                break;
            case 3:
                Screen.SetResolution(2560, 1440, Screen.fullScreen); // Option 4: 2560x1440
                break;
            case 4:
                Screen.SetResolution(3840, 2160, Screen.fullScreen); // Option 5: 3840x2160
                break;
            // Add more cases for additional resolutions if needed
        }
    }
}
