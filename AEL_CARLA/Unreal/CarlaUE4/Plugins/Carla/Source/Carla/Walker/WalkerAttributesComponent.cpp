// Copyright (c) 2024 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "WalkerAttributesComponent.h"

#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"

UWalkerAttributesComponent::UWalkerAttributesComponent()
{
  PrimaryComponentTick.bCanEverTick = false;
}

void UWalkerAttributesComponent::BeginPlay()
{
  Super::BeginPlay();

  // Automatically apply speed multiplier when the component begins play
  ApplySpeedMultiplier();
}

void UWalkerAttributesComponent::ApplySpeedMultiplier()
{
  AActor* Owner = GetOwner();
  if (!Owner)
  {
    return;
  }

  ACharacter* Character = Cast<ACharacter>(Owner);
  if (!Character)
  {
    return;
  }

  UCharacterMovementComponent* MovementComponent = Character->GetCharacterMovement();
  if (!MovementComponent)
  {
    return;
  }

  // Apply the speed multiplier to the character's max walk speed
  MovementComponent->MaxWalkSpeed *= SpeedMultiplier;
}

void UWalkerAttributesComponent::SetCustomAttribute(const FString& Key, const FString& Value)
{
  CustomAttributes.Add(Key, Value);
}

FString UWalkerAttributesComponent::GetCustomAttribute(const FString& Key) const
{
  const FString* Value = CustomAttributes.Find(Key);
  return Value ? *Value : FString();
}

bool UWalkerAttributesComponent::HasCustomAttribute(const FString& Key) const
{
  return CustomAttributes.Contains(Key);
}

bool UWalkerAttributesComponent::RemoveCustomAttribute(const FString& Key)
{
  return CustomAttributes.Remove(Key) > 0;
}
