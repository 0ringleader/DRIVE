using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class UIButtonHandler : MonoBehaviour
{
    public Material groundMaterial;
    public Texture groundTexture;
    public CustomCarController carController;
    public MJPEGStream mjpegStream;
    private TextMeshProUGUI fpsText; // Referenz für FPS-Anzeige
    private TextMeshProUGUI mouseWheelText;
    private TextMeshProUGUI speedText;
    private TextMeshProUGUI steeringText;
    private Slider gameSpeedSlider;
    private TextMeshProUGUI gameSpeedText;
    private TMP_Dropdown resolutionDropdown;
    private float deltaTime = 0.0f;

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
        syncToGameSpeedToggle.onValueChanged.AddListener(delegate { ToggleSyncToGameSpeed(syncToGameSpeedToggle); });
        gameSpeedSlider.onValueChanged.AddListener(UpdateGameSpeed);
        resolutionDropdown.onValueChanged.AddListener(delegate { ChangeResolution(resolutionDropdown); });

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
    }

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

        if (gameSpeedText != null)
        {
            gameSpeedText.text = $"Game Speed: {newSpeed:F2}x";
        }
        if (mjpegStream != null)
        {
            mjpegStream.setGameSpeed(newSpeed);
        }
    }

    void ChangeResolution(TMP_Dropdown dropdown)
    {
        SetResolution(dropdown.value);
    }
}
