## ADDED Requirements

### Requirement: Configurable Stage2 EMA teacher
The system SHALL provide a configuration-controlled EMA teacher for Stage2 training that is disabled by default and initialized from the student model when enabled.

#### Scenario: EMA teacher disabled
- **WHEN** the Stage2 image-level EMA KD option is disabled
- **THEN** training SHALL follow the existing Stage2 source-supervised plus GRL domain-adversarial behavior without creating teacher outputs or image-level KD losses

#### Scenario: EMA teacher enabled
- **WHEN** the Stage2 image-level EMA KD option is enabled
- **THEN** training SHALL initialize a teacher model from the loaded student weights, run teacher inference without gradients, and update teacher weights from student weights using EMA

### Requirement: Strong target view without geometric transforms
The system SHALL create a strong target view using Adaptive Teacher-style photometric and occlusion augmentations without random crop, resize crop, horizontal flip, or any transform that requires box coordinate remapping.

#### Scenario: Target strong augmentation
- **WHEN** a target-domain training image is used for image-level KD
- **THEN** the student strong view SHALL be derived from the same geometry as the teacher view and may apply color jitter, random grayscale, Gaussian blur, and cutout according to configuration

### Requirement: H2FA image-level aggregation
The system SHALL aggregate per-image ROI classification logits and offline RPN objectness logits into class-wise image-level probabilities using the H2FA aggregation formula.

#### Scenario: Aggregating one target image
- **WHEN** ROI logits and proposal objectness logits are available for a target image
- **THEN** the system SHALL exclude background logits, build class-wise objectness by assigning each proposal objectness to its highest-logit class, apply row-wise softmax over classes and column-wise softmax over proposals, and sum over proposals to produce one probability per class

### Requirement: Image-level KD loss
The system SHALL train the student recognition branch on target images with an optional image-level consistency loss between student strong-view probabilities and detached teacher weak-view probabilities.

#### Scenario: KD warmup not reached
- **WHEN** the current training iteration is below the configured image-level KD warmup
- **THEN** the system SHALL skip the image-level KD loss while continuing source supervised and GRL losses

#### Scenario: KD warmup reached
- **WHEN** the current training iteration is at or above the configured image-level KD warmup
- **THEN** the system SHALL add a weighted image-level multi-label consistency loss to the existing Stage2 losses

### Requirement: Frozen offline localization branch
The system SHALL keep the offline RPN/backbone frozen when image-level EMA KD is enabled.

#### Scenario: Offline branch during KD training
- **WHEN** image-level EMA KD training is active
- **THEN** offline RPN/backbone parameters SHALL remain excluded from optimizer updates and their objectness logits SHALL be used only as aggregation inputs
