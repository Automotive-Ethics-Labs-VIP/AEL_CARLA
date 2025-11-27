// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/Walker/WalkerAttributesComponent.h"

UWalkerAttributesComponent::UWalkerAttributesComponent()
{
  PrimaryComponentTick.bCanEverTick = false;
}

void UWalkerAttributesComponent::SetElderlyDefaults()
{
  bIsElderly = true;
  SpeedMultiplier = 0.6f;
}

float UWalkerAttributesComponent::GetEffectiveSpeedMultiplier() const
{
  // If SpeedMultiplier has been explicitly set (not default), use it directly
  if (SpeedMultiplier != 1.0f)
  {
    return SpeedMultiplier;
  }

  // Otherwise, compute based on attribute flags
  // These multipliers can be tuned based on realistic walking speeds
  if (bIsElderly)
  {
    return 0.6f;  // Elderly walk at ~60% normal speed
  }
  if (bIsChild)
  {
    return 0.7f;  // Children walk at ~70% normal speed (shorter stride)
  }
  if (bIsDisabled)
  {
    return 0.5f;  // Disabled pedestrians walk at ~50% normal speed
  }
  if (bIsPregnant)
  {
    return 0.8f;  // Pregnant pedestrians walk at ~80% normal speed
  }

  return SpeedMultiplier;
}
