// Author: Felix Kirch
// Date: 23.08.2024
// Description: Prevents the object from being destroyed when loading a new scene to optimize track switiching

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PersistentObject : MonoBehaviour
{
    private void Awake()
    {
        DontDestroyOnLoad(gameObject); // prevents the object from being destroyed when loading a new scene
    }
}

