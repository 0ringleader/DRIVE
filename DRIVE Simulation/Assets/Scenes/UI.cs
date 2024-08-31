using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using Application = UnityEngine.Application;
using Random = UnityEngine.Random;

public class UIButtonHandler : MonoBehaviour
{
    public Material groundMaterial;
    public Texture groundTexture;
    public CustomCarController carController;
    public MJPEGStream mjpegStream;
    
    // UI elements
    private TextMeshProUGUI fpsText;
    private TextMeshProUGUI mouseWheelText;
    private TextMeshProUGUI speedText;
    private TextMeshProUGUI steeringText;
    private Slider gameSpeedSlider;
    private TextMeshProUGUI gameSpeedText;
    private TMP_Dropdown resolutionDropdown;
    private Toggle autoSceneSwitchToggle;
    private Button randomSceneButton;
    private Toggle toggleTextureToggle;
    private Button exitButton;
    private Button resetCarButton;
    private Toggle controlByWebsiteToggle;
    private Toggle toggleOffRoadReset;
    private Toggle syncToGameSpeedToggle;
    private TextMeshProUGUI offRoadWarning;
    private Toggle randStartDirToggle;
    
    private float deltaTime = 0.0f; // For FPS calculation
    private List<string> trackScenes = new List<string> // List of available track scenes
    {
        "Track1",
        "Track2",
        "Track3",
        "Track4",
        "Track5",
        "Track6",
        "Track7",
        "Track8",
        "Track9"
    };

    private bool isUIVisible = true;

    void Start()
    {
        // Find the UI elements in the scene
        mouseWheelText = GameObject.Find("MousewheelText").GetComponent<TextMeshProUGUI>();
        speedText = GameObject.Find("SpeedText").GetComponent<TextMeshProUGUI>();
        steeringText = GameObject.Find("SteeringText").GetComponent<TextMeshProUGUI>();
        gameSpeedSlider = GameObject.Find("GameSpeedSlider").GetComponent<Slider>();
        gameSpeedText = GameObject.Find("GameSpeedText").GetComponent<TextMeshProUGUI>();
        resolutionDropdown = GameObject.Find("ResolutionDropdown").GetComponent<TMP_Dropdown>();
        fpsText = GameObject.Find("fpsText").GetComponent<TextMeshProUGUI>();
        autoSceneSwitchToggle = GameObject.Find("AutoSceneSwitchToggle").GetComponent<Toggle>();
        randomSceneButton = GameObject.Find("RandomSceneButton").GetComponent<Button>();
        toggleTextureToggle = GameObject.Find("ToggleTextureToggle").GetComponent<Toggle>();
        exitButton = GameObject.Find("ExitButton").GetComponent<Button>();
        resetCarButton = GameObject.Find("ResetCarButton").GetComponent<Button>();
        controlByWebsiteToggle = GameObject.Find("ControlByWebsiteToggle").GetComponent<Toggle>();
        toggleOffRoadReset = GameObject.Find("ToggleOffRoadReset").GetComponent<Toggle>();
        syncToGameSpeedToggle = GameObject.Find("SyncToGameSpeedToggle").GetComponent<Toggle>();
        offRoadWarning = GameObject.Find("OffRoadWarning").GetComponent<TextMeshProUGUI>();
        randStartDirToggle = GameObject.Find("RandStartDirToggle").GetComponent<Toggle>();

        // Add listeners to the UI elements
        toggleTextureToggle.onValueChanged.AddListener(delegate { ToggleGroundTexture(toggleTextureToggle); });
        exitButton.onClick.AddListener(ExitApplication);
        resetCarButton.onClick.AddListener(ResetCar);
        controlByWebsiteToggle.onValueChanged.AddListener(delegate { ToggleControlByWebsite(controlByWebsiteToggle); });
        toggleOffRoadReset.onValueChanged.AddListener(delegate { ToggleOffRoadReset(toggleOffRoadReset); });
        syncToGameSpeedToggle.onValueChanged.AddListener(delegate { ToggleSyncToGameSpeed(syncToGameSpeedToggle); });
        gameSpeedSlider.onValueChanged.AddListener(UpdateGameSpeed);
        resolutionDropdown.onValueChanged.AddListener(delegate { ChangeResolution(resolutionDropdown); });
        randomSceneButton.onClick.AddListener(SwitchToRandomScene);
        autoSceneSwitchToggle.onValueChanged.AddListener(delegate { ToggleAutoSceneSwitch(autoSceneSwitchToggle); });
        randStartDirToggle.onValueChanged.AddListener(delegate { carController.randomizeStartingDirection = randStartDirToggle.isOn; });
        
        // Initialize UI elements
        PopulateResolutionDropdown();
        ToggleMouseWheelText(controlByWebsiteToggle.isOn);
        UpdateGameSpeed(gameSpeedSlider.value);
    }

    void Update()
    {
        // Update FPS
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        int fps = Mathf.RoundToInt(1.0f / deltaTime);

        if (fpsText != null)
        {
            fpsText.text = $"FPS: {fps}";
        }

        // Exit application when Escape key is pressed
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ExitApplication();
        }

        // Toggle UI visibility when 'H' key is pressed
        if (Input.GetKeyDown(KeyCode.H))
        {
            ToggleUIVisibility();
        }
    }

    // Toggles the visibility of the UI
    void ToggleUIVisibility()
    {
        isUIVisible = !isUIVisible;

        mouseWheelText.gameObject.SetActive(isUIVisible);
        speedText.gameObject.SetActive(isUIVisible);
        steeringText.gameObject.SetActive(isUIVisible);
        gameSpeedSlider.gameObject.SetActive(isUIVisible);
        gameSpeedText.gameObject.SetActive(isUIVisible);
        resolutionDropdown.gameObject.SetActive(isUIVisible);
        fpsText.gameObject.SetActive(isUIVisible);
        autoSceneSwitchToggle.gameObject.SetActive(isUIVisible);
        randomSceneButton.gameObject.SetActive(isUIVisible);
        toggleTextureToggle.gameObject.SetActive(isUIVisible);
        exitButton.gameObject.SetActive(isUIVisible);
        resetCarButton.gameObject.SetActive(isUIVisible);
        controlByWebsiteToggle.gameObject.SetActive(isUIVisible);
        toggleOffRoadReset.gameObject.SetActive(isUIVisible);
        syncToGameSpeedToggle.gameObject.SetActive(isUIVisible);
        offRoadWarning.gameObject.SetActive(isUIVisible);
    }

    // Adds the available resolutions to the dropdown menu
    void PopulateResolutionDropdown()
    {
        resolutionDropdown.ClearOptions();
        Resolution[] resolutions = Screen.resolutions;
        List<string> options = new List<string>();

        foreach (Resolution res in resolutions)
        {
            string option = res.width + " x " + res.height;
            options.Add(option);
        }

        resolutionDropdown.AddOptions(options);
    }

    // Sets the screen resolution
    void SetResolution(int resolutionIndex)
    {
        Resolution resolution = Screen.resolutions[resolutionIndex];
        Screen.SetResolution(resolution.width, resolution.height, Screen.fullScreen);
    }

    // Toggles the ground texture on or off in the case the ground confuses the algorithm
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

    // Exits the application
    void ExitApplication()
    {
        Application.Quit();
    }

    // Resets the car's position and orientation
    void ResetCar()
    {
        if (carController != null)
        {
            carController.ResetCar();
        }
    }

    // Toggles control of the car via JSON input
    void ToggleControlByWebsite(Toggle toggle)
    {
        if (carController != null)
        {
            carController.isControlledByWebsite = toggle.isOn;
        }
        ToggleMouseWheelText(toggle.isOn);
    }

    // Updates the control tips
    void ToggleMouseWheelText(bool isControlledByWebsite)
    {
        if (mouseWheelText != null)
        {
            mouseWheelText.text = isControlledByWebsite ? "" : "Control speed with mousewheel or arrow keys";
        }
    }

    // Toggles whether the car should reset when it goes off-road
    void ToggleOffRoadReset(Toggle toggle)
    {
        if (carController != null)
        {
            carController.resetOnOffRoad = toggle.isOn;
        }
    }

    // Toggles syncing the stream's frame rate with the simulation speed to send more data to the training networks accordingly
    void ToggleSyncToGameSpeed(Toggle toggle)
    {
        if (mjpegStream != null)
        {
            mjpegStream.syncToGameSpeed = toggle.isOn;
        }
    }

    // Toggles automatic scene switching
    void ToggleAutoSceneSwitch(Toggle toggle)
    {
        carController.autoSwitchScenes = toggle.isOn;
    }

    // Updates the speed text displayed in the UI
    public void UpdateSpeedText(float speed, float mappedSpeed)
    {
        if (speedText != null)
        {
            speedText.text = $"Speed: {speed:F2}m/s, value: {mappedSpeed:F0}";
        }
    }

    // Updates the steering text displayed in the UI
    public void UpdateSteeringText(float steering, float mappedSteering)
    {
        if (steeringText != null)
        {
            steeringText.text = $"Steering: {steering:F2}°, value: {mappedSteering:F0}";
        }
    }

    // Updates the simulation speed and reflects the change in the UI and stream
    void UpdateGameSpeed(float newSpeed)
    {
        newSpeed = newSpeed / 10f;
        Time.timeScale = newSpeed;

        if (gameSpeedText != null)
        {
            gameSpeedText.text = $"Game Speed: {newSpeed:F2}x";
        }
        if (mjpegStream != null)
        {
            mjpegStream.setGameSpeed(newSpeed);
        }
    }

    // Changes the screen resolution based on the dropdown selection
    void ChangeResolution(TMP_Dropdown dropdown)
    {
        SetResolution(dropdown.value);
    }

    // Switches to a random track scene
    public void SwitchToRandomScene()
    {
        if (trackScenes.Count == 0)
        {
            Debug.LogWarning("No track scenes found.");
            return;
        }

        carController.ResetCar();
        int randomIndex = Random.Range(0, trackScenes.Count);
        SceneManager.LoadScene(trackScenes[randomIndex]);
    }
}
