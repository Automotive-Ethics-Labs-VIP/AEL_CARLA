// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "WalkerAttributesComponent.generated.h"

/// Component that stores custom pedestrian attributes at runtime.
/// Allows creating elderly, child, pregnant, or disabled pedestrian variants
/// without modifying Blueprint assets.
UCLASS(Blueprintable, BlueprintType, ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class CARLA_API UWalkerAttributesComponent : public UActorComponent
{
  GENERATED_BODY()

public:

  UWalkerAttributesComponent();

  // ===========================================================================
  /// @name Attribute Flags
  // ===========================================================================
  /// @{

  /// Whether this pedestrian is elderly
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsElderly = false;

  /// Whether this pedestrian is a child
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsChild = false;

  /// Whether this pedestrian is pregnant
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsPregnant = false;

  /// Whether this pedestrian is disabled
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsDisabled = false;

  /// @}
  // ===========================================================================
  /// @name Speed Control
  // ===========================================================================
  /// @{

  /// Multiplier applied to walker speed (1.0 = normal speed)
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "2.0"))
  float SpeedMultiplier = 1.0f;

  /// @}
  // ===========================================================================
  /// @name Custom Tags
  // ===========================================================================
  /// @{

  /// Custom key-value tags for extensibility
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  TMap<FString, FString> CustomTags;

  /// @}
  // ===========================================================================
  /// @name Helper Functions
  // ===========================================================================
  /// @{

  /// Sets default values for an elderly pedestrian (slower walking speed)
  UFUNCTION(BlueprintCallable, Category = "Walker Attributes")
  void SetElderlyDefaults();

  /// Gets the effective speed multiplier for this walker
  UFUNCTION(BlueprintCallable, Category = "Walker Attributes")
  float GetEffectiveSpeedMultiplier() const;

  /// @}
};
