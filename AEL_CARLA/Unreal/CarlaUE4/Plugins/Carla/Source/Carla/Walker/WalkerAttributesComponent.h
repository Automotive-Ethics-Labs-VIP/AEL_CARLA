// Copyright (c) 2024 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"

#include "WalkerAttributesComponent.generated.h"

/// Component for attaching custom attributes to walker actors at spawn time.
/// Allows creating pedestrian variants (e.g., elderly, child, pregnant, disabled)
/// without editing binary Blueprint assets.
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class CARLA_API UWalkerAttributesComponent : public UActorComponent
{
  GENERATED_BODY()

public:

  UWalkerAttributesComponent();

  /// Whether this walker represents an elderly pedestrian
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsElderly = false;

  /// Whether this walker represents a child pedestrian
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsChild = false;

  /// Whether this walker represents a pregnant pedestrian
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsPregnant = false;

  /// Whether this walker represents a disabled pedestrian
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  bool bIsDisabled = false;

  /// Multiplier applied to the walker's base movement speed.
  /// Values less than 1.0 result in slower walking (e.g., 0.5 for elderly).
  /// Values greater than 1.0 result in faster walking.
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.1", ClampMax = "2.0"))
  float SpeedMultiplier = 1.0f;

  /// Custom attributes map for extensibility.
  /// Allows attaching arbitrary key-value pairs to walkers at runtime.
  UPROPERTY(Category = "Walker Attributes", EditAnywhere, BlueprintReadWrite)
  TMap<FString, FString> CustomAttributes;

  /// Apply the speed multiplier to the owning walker's movement component.
  UFUNCTION(Category = "Walker Attributes", BlueprintCallable)
  void ApplySpeedMultiplier();

  /// Set a custom attribute by key.
  UFUNCTION(Category = "Walker Attributes", BlueprintCallable)
  void SetCustomAttribute(const FString& Key, const FString& Value);

  /// Get a custom attribute by key. Returns empty string if not found.
  UFUNCTION(Category = "Walker Attributes", BlueprintCallable, BlueprintPure)
  FString GetCustomAttribute(const FString& Key) const;

  /// Check if a custom attribute exists.
  UFUNCTION(Category = "Walker Attributes", BlueprintCallable, BlueprintPure)
  bool HasCustomAttribute(const FString& Key) const;

  /// Remove a custom attribute by key. Returns true if the attribute was removed.
  UFUNCTION(Category = "Walker Attributes", BlueprintCallable)
  bool RemoveCustomAttribute(const FString& Key);

protected:

  virtual void BeginPlay() override;

};
